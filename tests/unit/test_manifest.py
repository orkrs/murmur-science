"""Tests for dataset manifest."""


from murmur.data.manifest import DatasetManifest, compute_shard_hash


class TestManifest:
    """Test manifest creation and serialization."""

    def test_manifest_to_dict(self):
        """Manifest converts to dictionary."""
        manifest = DatasetManifest(
            dataset_name="test",
            revision="v1.0",
            license="Apache-2.0",
            source="test corpus",
            filters={"language": "en", "min_length": 100},
            tokenizer_fingerprint="abc123",
            shard_hashes=["hash1", "hash2"],
            num_documents=1000,
            num_tokens=500000,
            num_accepted=950,
            num_rejected=50,
            created_at="2026-01-01T00:00:00Z",
        )
        data = manifest.to_dict()
        assert data["dataset_name"] == "test"
        assert data["num_tokens"] == 500000

    def test_manifest_json_roundtrip(self, tmp_path):
        """Manifest round-trips through JSON."""
        manifest = DatasetManifest(
            dataset_name="test",
            revision="v1.0",
            license="Apache-2.0",
            source="test corpus",
            filters={},
            tokenizer_fingerprint="abc123",
            shard_hashes=["hash1"],
            num_documents=100,
            num_tokens=50000,
            num_accepted=100,
            num_rejected=0,
            created_at="2026-01-01T00:00:00Z",
        )
        json_path = tmp_path / "manifest.json"
        manifest.to_json(json_path)
        loaded = DatasetManifest.from_json(json_path)
        assert loaded.dataset_name == manifest.dataset_name
        assert loaded.num_tokens == manifest.num_tokens

    def test_manifest_fingerprint_stability(self):
        """Same manifest produces same fingerprint."""
        manifest1 = DatasetManifest(
            dataset_name="test",
            revision="v1.0",
            license="Apache-2.0",
            source="test",
            filters={},
            tokenizer_fingerprint="abc",
            shard_hashes=["h1"],
            num_documents=100,
            num_tokens=50000,
            num_accepted=100,
            num_rejected=0,
            created_at="2026-01-01T00:00:00Z",
        )
        manifest2 = DatasetManifest(
            dataset_name="test",
            revision="v1.0",
            license="Apache-2.0",
            source="test",
            filters={},
            tokenizer_fingerprint="abc",
            shard_hashes=["h1"],
            num_documents=100,
            num_tokens=50000,
            num_accepted=100,
            num_rejected=0,
            created_at="2026-01-02T00:00:00Z",
        )
        assert manifest1.fingerprint() == manifest2.fingerprint()


class TestShardHash:
    """Test shard hash computation."""

    def test_compute_shard_hash(self, tmp_path):
        """Compute SHA-256 hash of shard file."""
        shard_path = tmp_path / "shard.bin"
        shard_path.write_bytes(b"test data")
        h = compute_shard_hash(shard_path)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
