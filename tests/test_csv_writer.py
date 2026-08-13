"""
Comprehensive tests for CSVWriterThread class.

Tests cover:
- Basic initialization and header writing
- Thread-safe enqueueing and writing
- Graceful shutdown with pending data
- File integrity after completion
"""

import csv
import tempfile
import time
from pathlib import Path

from hydra_suite.data.csv_writer import CSVWriterThread


class TestCSVWriterThread:
    """Test suite for CSVWriterThread class."""

    def test_initialization_with_header(self):
        """Test that CSVWriterThread initializes correctly with a header."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_path = f.name

        try:
            header = ["TrackID", "X", "Y", "FrameID"]
            writer = CSVWriterThread(temp_path, header=header)
            writer.start()
            writer.stop()
            writer.join(timeout=2)

            # Verify file is created and header is written (after thread closes file)
            with open(temp_path, "r") as f:
                reader = csv.reader(f)
                first_row = next(reader)
                assert first_row == header
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_initialization_without_header(self):
        """Test that CSVWriterThread initializes correctly without a header."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_path = f.name

        try:
            writer = CSVWriterThread(temp_path)
            # Don't start the thread if we're not using it
            # Just verify initialization
            assert writer.csv_path == temp_path
            assert writer.header == []
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_single_row_write(self):
        """Test writing a single row of data."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_path = f.name

        try:
            header = ["TrackID", "X", "Y", "FrameID"]
            writer = CSVWriterThread(temp_path, header=header)
            writer.start()

            # Enqueue a single row
            row_data = [0, 100.5, 200.3, 1]
            writer.enqueue(row_data)

            # Give thread time to process
            time.sleep(0.5)

            # Stop and wait for completion
            writer.stop()
            writer.join(timeout=2)

            # Verify data was written
            with open(temp_path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)
                assert len(rows) == 2  # Header + 1 data row
                assert rows[0] == header
                assert rows[1] == [str(x) for x in row_data]
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_multiple_rows_write(self):
        """Test writing multiple rows of data."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_path = f.name

        try:
            header = ["TrackID", "X", "Y", "FrameID"]
            writer = CSVWriterThread(temp_path, header=header)
            writer.start()

            # Enqueue multiple rows
            test_data = [
                [0, 100.5, 200.3, 1],
                [1, 150.2, 250.8, 1],
                [0, 105.1, 203.7, 2],
                [1, 155.9, 248.2, 2],
            ]

            for row in test_data:
                writer.enqueue(row)

            # Give thread time to process
            time.sleep(0.5)

            # Stop and wait for completion
            writer.stop()
            writer.join(timeout=2)

            # Verify all data was written
            with open(temp_path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)
                assert len(rows) == len(test_data) + 1  # Header + data rows
                assert rows[0] == header
                for i, data_row in enumerate(test_data):
                    assert rows[i + 1] == [str(x) for x in data_row]
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_high_volume_writes(self):
        """Test writing a large volume of data (stress test)."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_path = f.name

        try:
            header = ["TrackID", "X", "Y", "FrameID"]
            writer = CSVWriterThread(temp_path, header=header)
            writer.start()

            # Enqueue 1000 rows rapidly
            num_rows = 1000
            test_data = [[i % 10, i * 1.5, i * 2.3, i] for i in range(num_rows)]

            for row in test_data:
                writer.enqueue(row)

            # Stop and wait for completion
            writer.stop()
            writer.join(timeout=5)

            # Verify all data was written
            with open(temp_path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)
                assert len(rows) == num_rows + 1  # Header + data rows
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_thread_safety(self):
        """Test that multiple rapid enqueues don't corrupt data."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_path = f.name

        try:
            header = ["TrackID", "X", "Y", "FrameID"]
            writer = CSVWriterThread(temp_path, header=header)
            writer.start()

            # Rapidly enqueue data without waiting
            num_rows = 100
            for i in range(num_rows):
                writer.enqueue([i % 5, i * 1.1, i * 2.2, i])

            # Stop and wait for completion
            writer.stop()
            writer.join(timeout=3)

            # Verify all data was written in order
            with open(temp_path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)
                assert len(rows) == num_rows + 1
                for i in range(num_rows):
                    expected = [str(i % 5), str(i * 1.1), str(i * 2.2), str(i)]
                    assert rows[i + 1] == expected
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_graceful_shutdown_with_pending_data(self):
        """Test that stopping the thread flushes all pending data."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_path = f.name

        try:
            header = ["TrackID", "X", "Y", "FrameID"]
            writer = CSVWriterThread(temp_path, header=header)
            writer.start()

            # Enqueue data and immediately stop (data should still be written)
            num_rows = 50
            for i in range(num_rows):
                writer.enqueue([i, i * 10, i * 20, i])

            # Stop immediately without waiting
            writer.stop()
            writer.join(timeout=3)

            # Verify all data was flushed before shutdown
            with open(temp_path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)
                assert len(rows) == num_rows + 1  # All data should be written
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_empty_queue_stop(self):
        """Test stopping thread when queue is empty."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_path = f.name

        try:
            header = ["TrackID", "X", "Y"]
            writer = CSVWriterThread(temp_path, header=header)
            writer.start()

            # Stop immediately without enqueueing any data
            writer.stop()
            writer.join(timeout=2)

            # Verify file has only header
            with open(temp_path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)
                assert len(rows) == 1
                assert rows[0] == header
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_mixed_data_types(self):
        """Test writing mixed data types (int, float, string)."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_path = f.name

        try:
            header = ["TrackID", "X", "Y", "State", "FrameID"]
            writer = CSVWriterThread(temp_path, header=header)
            writer.start()

            # Mix of integers, floats, and strings
            test_data = [
                [0, 100.5, 200.3, "active", 1],
                [1, 150.2, 250.8, "occluded", 1],
                [0, 105.1, 203.7, "lost", 2],
            ]

            for row in test_data:
                writer.enqueue(row)

            time.sleep(0.5)
            writer.stop()
            writer.join(timeout=2)

            # Verify mixed types were written correctly
            with open(temp_path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)
                assert len(rows) == 4
                for i, data_row in enumerate(test_data):
                    assert rows[i + 1] == [str(x) for x in data_row]
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_queue_property(self):
        """Test that queue is accessible and functional."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_path = f.name

        try:
            writer = CSVWriterThread(temp_path)
            assert writer.queue.empty()  # Should start empty

            writer.enqueue([1, 2, 3])
            assert not writer.queue.empty()  # Should have data now

            writer.start()
            time.sleep(0.5)
            writer.stop()
            writer.join(timeout=2)

            # After processing, queue should be empty again
            assert writer.queue.empty()
        finally:
            Path(temp_path).unlink(missing_ok=True)
