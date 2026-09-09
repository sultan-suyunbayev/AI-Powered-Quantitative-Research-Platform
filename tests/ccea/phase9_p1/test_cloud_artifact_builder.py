# -*- coding: utf-8 -*-
from __future__ import annotations

import json

from ccea.crypto.keys import KeyAlgorithm, generate_keypair
from packages.cloud.builder.artifact_builder import ArtifactBuilder, BuildConfig
from packages.cloud.builder.artifact_publisher import (
    ArtifactPublisher,
    PublishStatus,
    RegistryConfig,
    RegistryType,
)
from packages.shared.contracts.manifest import ArtifactFormat
from packages.shared.utils.hashing import compute_file_hash


def test_cloud_artifact_builder_blocks_unsigned_artifacts(tmp_path):
    signing_key = generate_keypair(KeyAlgorithm.ED25519, key_id="test-key")
    builder = ArtifactBuilder(signing_key=signing_key)

    src = tmp_path / "src"
    src.mkdir()
    (src / "strategy.py").write_text("class S: pass\n", encoding="utf-8")

    result = builder.build(
        BuildConfig(
            strategy_id="s1",
            strategy_name="Strategy 1",
            version="1.0.0",
            entrypoint="strategy:S",
            source_path=src,
            output_format=ArtifactFormat.ZIP_BUNDLE,
            output_path=tmp_path / "out",
            sign_artifact=False,
        )
    )

    assert result.success is False
    assert any("Unsigned artifacts are not allowed" in e for e in result.errors)


def test_cloud_artifact_builder_emits_signature_sbom_and_provenance(tmp_path):
    signing_key = generate_keypair(KeyAlgorithm.ED25519, key_id="test-key")
    builder = ArtifactBuilder(
        registry_url="registry.test", builder_id="builder-test", signing_key=signing_key
    )

    src = tmp_path / "src"
    src.mkdir()
    (src / "strategy.py").write_text("class Strategy: pass\n", encoding="utf-8")

    req = tmp_path / "requirements.txt"
    req.write_text("requests==2.31.0\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    result = builder.build(
        BuildConfig(
            strategy_id="s1",
            strategy_name="Strategy 1",
            version="1.2.3",
            entrypoint="strategy:Strategy",
            source_path=src,
            requirements_file=req,
            output_format=ArtifactFormat.ZIP_BUNDLE,
            output_path=out_dir,
        )
    )

    assert result.success is True
    assert result.artifact_path == out_dir
    assert result.artifact_file == out_dir / "artifact.zip"
    assert (out_dir / "artifact.zip").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "sbom.cyclonedx.json").exists()

    assert result.manifest is not None
    assert result.manifest.has_valid_signature() is True
    assert result.manifest.sbom_ref == "sbom.cyclonedx.json"
    assert result.manifest.provenance.builder_id == "builder-test"
    assert result.manifest.provenance.build_host

    expected_lock_digest = compute_file_hash(req, with_prefix=True)
    assert result.manifest.provenance.deps_lock_digest == expected_lock_digest

    # Manifest file matches contract serialization
    manifest_json = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_json["artifact_digest"] == result.artifact_digest
    assert manifest_json["signature"]["signature_value"]
    assert manifest_json["sbom_ref"] == "sbom.cyclonedx.json"


def test_build_result_feeds_the_publisher(tmp_path):
    """The two halves compose: what build() produced is what publish() takes.

    artifact_path is the bundle directory -- artifact, manifest, SBOM -- and
    publish() opens a file and hashes it against manifest.artifact_digest. The
    file is artifact_file; handing over the directory instead fails on open().
    """
    signing_key = generate_keypair(KeyAlgorithm.ED25519, key_id="test-key")
    builder = ArtifactBuilder(signing_key=signing_key)

    src = tmp_path / "src"
    src.mkdir()
    (src / "strategy.py").write_text("class Strategy: pass\n", encoding="utf-8")

    build = builder.build(
        BuildConfig(
            strategy_id="s1",
            strategy_name="Strategy 1",
            version="1.0.0",
            entrypoint="strategy:Strategy",
            source_path=src,
            output_format=ArtifactFormat.ZIP_BUNDLE,
            output_path=tmp_path / "out",
        )
    )
    assert build.success is True

    publisher = ArtifactPublisher(
        config=RegistryConfig(
            registry_type=RegistryType.LOCAL,
            registry_url=str(tmp_path / "registry"),
        )
    )
    published = publisher.publish(build.artifact_file, build.manifest)

    assert published.errors == []
    assert published.status == PublishStatus.PUBLISHED
    # The digest the publisher recomputed is the one the builder signed.
    assert published.artifact_digest == build.artifact_digest
