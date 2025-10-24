# ABOUTME: Unit tests for reference statement management system
# ABOUTME: Tests ReferenceStatementSet and ReferenceStatementManager with real file operations

import pytest

from src.ssr.reference_statements import (
    ReferenceStatementSet,
    ReferenceStatementManager,
    ReferenceSetVersion,
)


@pytest.mark.unit
class TestReferenceStatementSet:
    """Test suite for ReferenceStatementSet class."""

    def test_create_valid_reference_set(self, paper_reference_set_1):
        """Test creation of valid reference statement set."""
        ref_set = paper_reference_set_1

        assert ref_set.set_id == "set_1"
        assert len(ref_set.statements) == 5
        assert all(rating in ref_set.statements for rating in [1, 2, 3, 4, 5])

    def test_reference_set_statement_retrieval(self, paper_reference_set_1):
        """Test retrieving individual statements by rating."""
        ref_set = paper_reference_set_1

        statement_1 = ref_set.get_statement(1)
        statement_5 = ref_set.get_statement(5)

        assert "definitely not purchase" in statement_1.lower()
        assert "definitely purchase" in statement_5.lower()
        assert statement_1 != statement_5

    def test_reference_set_all_statements_retrieval(self, paper_reference_set_1):
        """Test retrieving all statements as ordered list."""
        ref_set = paper_reference_set_1

        all_statements = ref_set.get_all_statements()

        assert len(all_statements) == 5
        assert all(isinstance(stmt, str) for stmt in all_statements)
        assert all(len(stmt) > 0 for stmt in all_statements)

    def test_reference_set_metadata(self, paper_reference_set_1):
        """Test metadata storage and retrieval."""
        ref_set = paper_reference_set_1

        assert ref_set.metadata["source"] == "paper"
        assert ref_set.metadata["set_number"] == 1

    def test_reference_set_invalid_rating_raises_error(self, paper_reference_set_1):
        """Test that invalid rating raises KeyError."""
        ref_set = paper_reference_set_1

        with pytest.raises(KeyError):
            ref_set.get_statement(0)  # Invalid: below range

        with pytest.raises(KeyError):
            ref_set.get_statement(6)  # Invalid: above range

    def test_reference_set_equality(self):
        """Test equality comparison between reference sets."""
        set1 = ReferenceStatementSet(
            set_id="test",
            statements={1: "stmt1", 2: "stmt2", 3: "stmt3", 4: "stmt4", 5: "stmt5"},
            metadata={"version": 1},
        )

        set2 = ReferenceStatementSet(
            set_id="test",
            statements={1: "stmt1", 2: "stmt2", 3: "stmt3", 4: "stmt4", 5: "stmt5"},
            metadata={"version": 1},
        )

        set3 = ReferenceStatementSet(
            set_id="different",
            statements={1: "stmt1", 2: "stmt2", 3: "stmt3", 4: "stmt4", 5: "stmt5"},
            metadata={"version": 1},
        )

        assert set1 == set2  # Same content
        assert set1 != set3  # Different ID


@pytest.mark.unit
class TestReferenceStatementManager:
    """Test suite for ReferenceStatementManager class."""

    def test_manager_initialization(self, test_data_directory):
        """Test manager initializes with storage directory."""
        manager = ReferenceStatementManager(
            storage_dir=test_data_directory / "ref_sets"
        )

        assert manager.storage_dir.exists()
        assert manager.storage_dir.is_dir()

    def test_add_and_retrieve_reference_set(
        self, test_data_directory, paper_reference_set_1
    ):
        """Test adding and retrieving a reference set."""
        manager = ReferenceStatementManager(
            storage_dir=test_data_directory / "test_ref_sets"
        )

        # Add reference set
        manager.add_set(paper_reference_set_1)

        # Retrieve and validate
        retrieved = manager.get_set("set_1")

        assert retrieved.set_id == paper_reference_set_1.set_id
        assert retrieved.statements == paper_reference_set_1.statements
        assert retrieved.metadata == paper_reference_set_1.metadata

    def test_add_multiple_reference_sets(
        self, test_data_directory, all_paper_reference_sets
    ):
        """Test managing multiple reference sets."""
        manager = ReferenceStatementManager(
            storage_dir=test_data_directory / "multi_sets"
        )

        # Add all 6 paper reference sets
        for ref_set in all_paper_reference_sets:
            manager.add_set(ref_set)

        # Verify all were added
        all_set_ids = manager.list_sets()

        assert len(all_set_ids) == 6
        assert all(f"set_{i}" in all_set_ids for i in range(1, 7))

    def test_retrieve_nonexistent_set_raises_error(self, test_data_directory):
        """Test retrieving non-existent set raises ValueError."""
        manager = ReferenceStatementManager(
            storage_dir=test_data_directory / "empty_sets"
        )

        with pytest.raises(ValueError, match="not found"):
            manager.get_set("nonexistent_set")

    def test_get_multiple_sets_by_ids(
        self, test_data_directory, all_paper_reference_sets
    ):
        """Test retrieving multiple sets at once."""
        manager = ReferenceStatementManager(
            storage_dir=test_data_directory / "batch_retrieval"
        )

        # Add all sets
        for ref_set in all_paper_reference_sets:
            manager.add_set(ref_set)

        # Retrieve first 3 sets
        retrieved_sets = manager.get_multiple_sets(["set_1", "set_2", "set_3"])

        assert len(retrieved_sets) == 3
        assert all(isinstance(s, ReferenceStatementSet) for s in retrieved_sets)
        assert [s.set_id for s in retrieved_sets] == ["set_1", "set_2", "set_3"]

    def test_update_existing_reference_set(
        self, test_data_directory, paper_reference_set_1
    ):
        """Test updating an existing reference set."""
        manager = ReferenceStatementManager(
            storage_dir=test_data_directory / "update_test"
        )

        # Add original
        manager.add_set(paper_reference_set_1)

        # Create updated version
        updated_set = ReferenceStatementSet(
            set_id="set_1",
            statements={
                1: "Updated statement 1",
                2: "Updated statement 2",
                3: "Updated statement 3",
                4: "Updated statement 4",
                5: "Updated statement 5",
            },
            metadata={"source": "paper", "set_number": 1, "updated": True},
        )

        # Update
        manager.add_set(updated_set, overwrite=True)

        # Retrieve and verify
        retrieved = manager.get_set("set_1")

        assert "Updated" in retrieved.statements[1]
        assert retrieved.metadata.get("updated") is True

    def test_delete_reference_set(self, test_data_directory, paper_reference_set_1):
        """Test deleting a reference set."""
        manager = ReferenceStatementManager(
            storage_dir=test_data_directory / "delete_test"
        )

        # Add and verify exists
        manager.add_set(paper_reference_set_1)
        assert "set_1" in manager.list_sets()

        # Delete
        manager.delete_set("set_1")

        # Verify removed
        assert "set_1" not in manager.list_sets()

        with pytest.raises(ValueError):
            manager.get_set("set_1")

    def test_file_persistence(self, test_data_directory, paper_reference_set_1):
        """Test reference sets persist across manager instances."""
        storage_dir = test_data_directory / "persistence_test"

        # Create manager and add set
        manager1 = ReferenceStatementManager(storage_dir=storage_dir)
        manager1.add_set(paper_reference_set_1)

        # Create new manager instance pointing to same directory
        manager2 = ReferenceStatementManager(storage_dir=storage_dir)

        # Verify set is available
        retrieved = manager2.get_set("set_1")

        assert retrieved.set_id == paper_reference_set_1.set_id
        assert retrieved.statements == paper_reference_set_1.statements

    def test_export_import_reference_set(
        self, test_data_directory, paper_reference_set_1
    ):
        """Test exporting and importing reference sets."""
        manager = ReferenceStatementManager(
            storage_dir=test_data_directory / "export_import"
        )

        # Add set
        manager.add_set(paper_reference_set_1)

        # Export to dict
        exported = manager.export_set("set_1")

        assert isinstance(exported, dict)
        assert exported["set_id"] == "set_1"
        assert "statements" in exported
        assert "metadata" in exported

        # Create new set from exported data
        imported_set = ReferenceStatementSet(
            set_id=exported["set_id"],
            statements={int(k): v for k, v in exported["statements"].items()},
            metadata=exported["metadata"],
        )

        # Verify equivalence
        assert imported_set == paper_reference_set_1


@pytest.mark.unit
class TestReferenceSetVersion:
    """Test suite for reference set versioning."""

    def test_version_creation(self):
        """Test creating a versioned reference set."""
        version = ReferenceSetVersion(
            set_id="test_set",
            version=1,
            statements={1: "v1", 2: "v1", 3: "v1", 4: "v1", 5: "v1"},
            metadata={"created_at": "2025-01-15"},
        )

        assert version.set_id == "test_set"
        assert version.version == 1
        assert version.get_full_id() == "test_set_v1"

    def test_version_comparison(self):
        """Test version comparison logic."""
        v1 = ReferenceSetVersion(
            set_id="test",
            version=1,
            statements={1: "s1", 2: "s2", 3: "s3", 4: "s4", 5: "s5"},
        )

        v2 = ReferenceSetVersion(
            set_id="test",
            version=2,
            statements={1: "s1", 2: "s2", 3: "s3", 4: "s4", 5: "s5"},
        )

        assert v2.version > v1.version
        assert v1.is_older_than(v2)
        assert v2.is_newer_than(v1)
