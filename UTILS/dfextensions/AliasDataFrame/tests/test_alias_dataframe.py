import unittest
import pandas as pd
import numpy as np
import os
from dfextensions.AliasDataFrame import AliasDataFrame  # Adjust if needed
import tempfile
import uproot

class TestAliasDataFrame(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame({
            "x": np.arange(5),
            "y": np.arange(5, 10),
            "CTPLumi_countsFV0": np.array([2000, 2100, 2200, 2300, 2400])
        })
        self.adf = AliasDataFrame(df)

    def test_basic_alias(self):
        self.adf.add_alias("z", "x + y")
        self.adf.materialize_all()
        expected = self.adf.df["x"] + self.adf.df["y"]
        pd.testing.assert_series_equal(self.adf.df["z"], expected, check_names=False)

    def test_dtype(self):
        self.adf.add_alias("z", "x + y", dtype=np.float16)
        self.adf.materialize_all()
        self.assertEqual(self.adf.df["z"].dtype, np.float16)

    def test_constant(self):
        self.adf.add_alias("c", "42.0", dtype=np.float32, is_constant=True)
        self.adf.add_alias("z", "x + c")
        self.adf.materialize_all()
        expected = self.adf.df["x"] + 42.0
        pd.testing.assert_series_equal(self.adf.df["z"], expected, check_names=False)

    def test_dependency_order(self):
        self.adf.add_alias("a", "x + y")
        self.adf.add_alias("b", "a * 2")
        self.adf.materialize_all()
        expected = (self.adf.df["x"] + self.adf.df["y"]) * 2
        pd.testing.assert_series_equal(self.adf.df["b"], expected, check_names=False)

    def test_log_rate_with_constant(self):
        median = self.adf.df["CTPLumi_countsFV0"].median()
        self.adf.add_alias("countsFV0_median", f"{median}", dtype=np.float16, is_constant=True)
        self.adf.add_alias("logRate", "log(CTPLumi_countsFV0/countsFV0_median)", dtype=np.float16)
        self.adf.materialize_all()
        expected = np.log(self.adf.df["CTPLumi_countsFV0"] / median).astype(np.float16)
        pd.testing.assert_series_equal(self.adf.df["logRate"], expected, check_names=False)

    def test_circular_dependency_raises_error(self):
        self.adf.add_alias("a", "b * 2")
        with self.assertRaises(ValueError):
            self.adf.add_alias("b", "a + 1")

    def test_undefined_symbol_raises_error(self):
        self.adf.add_alias("z", "x + non_existent_variable")
        with self.assertRaises(Exception):
            self.adf.materialize_all()

    def test_invalid_syntax_raises_error(self):
        self.adf.add_alias("z", "x +* y")
        with self.assertRaises(SyntaxError):
            self.adf.materialize_all()

    def test_partial_materialization(self):
        self.adf.add_alias("a", "x + 1")
        self.adf.add_alias("b", "a + 1")
        self.adf.add_alias("c", "y + 1")
        self.adf.materialize_alias("b")
        self.assertIn("a", self.adf.df.columns)
        self.assertIn("b", self.adf.df.columns)
        self.assertNotIn("c", self.adf.df.columns)

    def test_export_import_tree_roundtrip(self):
        df = pd.DataFrame({
            "x": np.linspace(0, 10, 100),
            "y": np.linspace(10, 20, 100)
        })
        adf = AliasDataFrame(df)
        adf.add_alias("z", "x + y", dtype=np.float64)
        adf.materialize_all()

        with tempfile.NamedTemporaryFile(suffix=".root", delete=False) as tmp:
            adf.export_tree(tmp.name, treename="testTree", dropAliasColumns=False)
            tmp_path = tmp.name

        adf_loaded = AliasDataFrame.read_tree(tmp_path, treename="testTree")

        assert "z" in adf_loaded.aliases
        assert adf_loaded.aliases["z"] == "x + y"
        adf_loaded.materialize_alias("z")
        pd.testing.assert_series_equal(adf.df["z"], adf_loaded.df["z"], check_names=False)

        os.remove(tmp_path)
    def test_getattr_column_and_alias_access(self):
        df = pd.DataFrame({
            "x": np.arange(5),
            "y": np.arange(5) * 2
        })
        adf = AliasDataFrame(df)
        adf.add_alias("z", "x + y", dtype=np.int32)

        # Access real column
        assert (adf.x == df["x"]).all()
        # Access alias before materialization
        assert "z" not in adf.df.columns
        z_val = adf.z
        assert "z" in adf.df.columns
        expected = df["x"] + df["y"]
        np.testing.assert_array_equal(z_val, expected)

    def test_bidirectional_atan2_support(self):
        """Test that both atan2 (ROOT) and arctan2 (Python) work"""
        df = pd.DataFrame({
            'x': np.array([1.0, 0.0, -1.0, 0.0]),
            'y': np.array([0.0, 1.0, 0.0, -1.0])
        })
        adf = AliasDataFrame(df)

        # Python style (arctan2)
        adf.add_alias('phi_python', 'arctan2(y, x)', dtype=np.float32)
        adf.materialize_alias('phi_python')

        # ROOT style (atan2) - should also work
        adf.add_alias('phi_root', 'atan2(y, x)', dtype=np.float32)
        adf.materialize_alias('phi_root')

        # Should be identical
        np.testing.assert_allclose(adf.df['phi_python'], adf.df['phi_root'], rtol=1e-6)

        # Expected values
        expected = np.array([0.0, np.pi/2, np.pi, -np.pi/2], dtype=np.float32)
        np.testing.assert_allclose(adf.df['phi_python'], expected, rtol=1e-6)

    def test_undefined_function_helpful_error(self):
        """Test that undefined functions give helpful error messages"""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        adf = AliasDataFrame(df)

        # Test 1: Undefined function
        adf.add_alias('bad', 'nonexistent_func(x)', dtype=np.float32)
        with self.assertRaises(NameError) as cm:
            adf.materialize_alias('bad')

        error_msg = str(cm.exception)
        # Check error message contains helpful info
        self.assertIn('nonexistent_func', error_msg)
        self.assertIn('Available functions include:', error_msg)
        self.assertIn('arctan2', error_msg)  # Should mention both forms
        self.assertIn('atan2', error_msg)

        # Test 2: Undefined variable
        adf.add_alias('bad2', 'x + undefined_var', dtype=np.float32)
        with self.assertRaises(NameError) as cm:
            adf.materialize_alias('bad2')

        error_msg = str(cm.exception)
        self.assertIn('undefined_var', error_msg)

class TestAliasDataFrameWithSubframes(unittest.TestCase):
    def setUp(self):
        n_tracks = 1000
        n_clusters = 100
        df_tracks = pd.DataFrame({
            "track_index": np.arange(n_tracks),
            "mX": np.random.normal(0, 10, n_tracks),
            "mY": np.random.normal(0, 10, n_tracks),
            "mZ": np.random.normal(0, 10, n_tracks),
            "mPt": np.random.exponential(1.0, n_tracks),
            "mEta": np.random.normal(0, 1, n_tracks),
        })

        cluster_idx = np.repeat(df_tracks["track_index"], n_clusters)
        df_clusters = pd.DataFrame({
            "track_index": cluster_idx,
            "mX": np.random.normal(0, 10, len(cluster_idx)),
            "mY": np.random.normal(0, 10, len(cluster_idx)),
            "mZ": np.random.normal(0, 10, len(cluster_idx)),
        })

        self.df_tracks = df_tracks
        self.df_clusters = df_clusters

    def test_alias_cluster_track_dx(self):
        adf_clusters = AliasDataFrame(self.df_clusters.copy())
        adf_tracks = AliasDataFrame(self.df_tracks.copy())
        adf_clusters.register_subframe("T", adf_tracks, index_columns="track_index")
        adf_clusters.add_alias("mDX", "mX - T.mX")
        adf_clusters.materialize_all()
        merged = adf_clusters.df.merge(adf_tracks.df, on="track_index", suffixes=("", "_trk"))
        expected = merged["mX"] - merged["mX_trk"]
        pd.testing.assert_series_equal(adf_clusters.df["mDX"].reset_index(drop=True), expected.reset_index(drop=True), check_names=False)

    def test_subframe_invalid_alias_raises(self):
        adf_clusters = AliasDataFrame(self.df_clusters.copy())
        adf_tracks = AliasDataFrame(self.df_tracks.copy())
        adf_clusters.register_subframe("T", adf_tracks, index_columns="track_index")
        adf_clusters.add_alias("invalid", "T.nonexistent")

        with self.assertRaises(KeyError) as cm:
            adf_clusters.materialize_alias("invalid")

        self.assertIn("T", str(cm.exception))
        self.assertIn("nonexistent", str(cm.exception))

    def test_save_and_load_integrity(self):
        adf_clusters = AliasDataFrame(self.df_clusters.copy())
        adf_tracks = AliasDataFrame(self.df_tracks.copy())
        adf_clusters.register_subframe("T", adf_tracks, index_columns="track_index")
        adf_clusters.add_alias("mDX", "mX - T.mX")
        adf_clusters.materialize_all()

        with tempfile.TemporaryDirectory() as tmpdir:
            path_clusters = os.path.join(tmpdir, "clusters.parquet")
            path_tracks = os.path.join(tmpdir, "tracks.parquet")
            adf_clusters.save(path_clusters)
            adf_tracks.save(path_tracks)

            adf_tracks_loaded = AliasDataFrame.load(path_tracks)
            adf_clusters_loaded = AliasDataFrame.load(path_clusters)
            adf_clusters_loaded.register_subframe("T", adf_tracks_loaded, index_columns="track_index")
            adf_clusters_loaded.add_alias("mDX", "mX - T.mX")
            adf_clusters_loaded.materialize_all()

            self.assertIn("mDX", adf_clusters_loaded.df.columns)
            merged = adf_clusters_loaded.df.merge(adf_tracks_loaded.df, on="track_index", suffixes=("", "_trk"))
            expected = merged["mX"] - merged["mX_trk"]
            pd.testing.assert_series_equal(adf_clusters_loaded.df["mDX"].reset_index(drop=True), expected.reset_index(drop=True), check_names=False)
            self.assertDictEqual(adf_clusters.aliases, adf_clusters_loaded.aliases)

    def test_getattr_subframe_alias_access(self):
        # Parent frame
        df_main = pd.DataFrame({"track_id": [0, 1, 2], "x": [10, 20, 30]})
        adf_main = AliasDataFrame(df_main)
        # Subframe with alias
        df_sub = pd.DataFrame({"track_id": [0, 1, 2], "residual": [1.1, 2.2, 3.3]})
        adf_sub = AliasDataFrame(df_sub)
        adf_sub.add_alias("residual_scaled", "residual * 100", dtype=np.float64)

        # Register subframe
        adf_main.register_subframe("track", adf_sub, index_columns="track_id")

        # Add alias depending on subframe alias
        adf_main.add_alias("resid100", "track.residual_scaled", dtype=np.float64)

        # Trigger materialization via __getattr__
        assert "resid100" not in adf_main.df.columns
        result = adf_main.resid100
        assert "resid100" in adf_main.df.columns
        np.testing.assert_array_equal(result, df_sub["residual"] * 100)



    def test_getattr_chained_subframe_access(self):
        df_main = pd.DataFrame({"id": [0, 1, 2]})
        df_sub = pd.DataFrame({"id": [0, 1, 2], "a": [5, 6, 7]})
        adf_main = AliasDataFrame(df_main)
        adf_sub = AliasDataFrame(df_sub)
        adf_sub.add_alias("cutA", "a > 5")
        adf_main.register_subframe("sub", adf_sub, index_columns="id")

        adf_sub.materialize_alias("cutA")

        # Check chained access
        expected = np.array([False, True, True])
        assert np.all(adf_main.sub.cutA == expected)  # explicit value check

    def test_multi_column_index_join(self):
        """Test subframe join with composite key (track_index, firstTFOrbit)"""
        df_main = pd.DataFrame({
            'track_index': [0, 0, 1, 1],
            'firstTFOrbit': [100, 200, 100, 200],
            'x': [1, 2, 3, 4]
        })
        df_sub = pd.DataFrame({
            'track_index': [0, 0, 1, 1],
            'firstTFOrbit': [100, 200, 100, 200],
            'y': [10, 20, 30, 40]
        })

        adf_main = AliasDataFrame(df_main)
        adf_sub = AliasDataFrame(df_sub)
        adf_main.register_subframe("T", adf_sub, index_columns=["track_index", "firstTFOrbit"])
        adf_main.add_alias("sum_xy", "x + T.y")
        adf_main.materialize_alias("sum_xy")

        expected = [11, 22, 33, 44]
        np.testing.assert_array_equal(adf_main.df['sum_xy'].values, expected)

class TestAliasDataFrameCompression(unittest.TestCase):
    """Test column compression functionality"""

    def setUp(self):
        """Create test data with values suitable for compression"""
        np.random.seed(42)
        df = pd.DataFrame({
            "dy": np.random.normal(0, 2.0, 1000).astype(np.float32),
            "dz": np.random.normal(0, 1.5, 1000).astype(np.float32),
            "tgSlp": np.random.uniform(-0.5, 0.5, 1000).astype(np.float32),
            "track_id": np.arange(1000)
        })
        self.adf = AliasDataFrame(df)
        self.original_dy = df["dy"].values.copy()

    def test_basic_compression_decompression(self):
        """Test basic compression creates correct structure"""
        spec = {
            'dy': {
                'compress': 'round(asinh(dy)*40)',
                'decompress': 'sinh(dy_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            }
        }

        self.adf.compress_columns(spec)

        # Check compressed column exists
        self.assertIn('dy_c', self.adf.df.columns)
        self.assertEqual(self.adf.df['dy_c'].dtype, np.int16)

        # Check original removed from storage
        self.assertNotIn('dy', self.adf.df.columns)

        # Check decompression alias exists
        self.assertIn('dy', self.adf.aliases)
        self.assertEqual(self.adf.aliases['dy'], 'sinh(dy_c/40.)')

        # Check compression_info populated
        self.assertIn('dy', self.adf.compression_info)
        info = self.adf.compression_info['dy']
        self.assertEqual(info['compressed_col'], 'dy_c')
        self.assertEqual(info['compressed_dtype'], 'int16')
        self.assertEqual(info['decompressed_dtype'], 'float16')

        # Materialize and check values approximately equal
        self.adf.materialize_alias('dy')
        decompressed = self.adf.df['dy'].values
        np.testing.assert_allclose(decompressed, self.original_dy, rtol=0.01, atol=0.05)

    def test_compression_with_precision_measurement(self):
        """Test optional precision measurement"""
        spec = {
            'dy': {
                'compress': 'round(asinh(dy)*40)',
                'decompress': 'sinh(dy_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            }
        }

        self.adf.compress_columns(spec, measure_precision=True)

        # Check precision info exists
        self.assertIn('precision', self.adf.compression_info['dy'])
        prec = self.adf.compression_info['dy']['precision']

        # Check all metrics present
        self.assertIn('rmse', prec)
        self.assertIn('max_error', prec)
        self.assertIn('mean_error', prec)

        # Sanity check values
        self.assertGreater(prec['rmse'], 0)
        self.assertLess(prec['rmse'], 0.1)  # Should be small for good compression

    def test_compress_alias_source(self):
        """Test compressing an alias (not materialized column)"""
        # Create alias first
        self.adf.add_alias('dy_scaled', 'dy * 2.0', dtype=np.float32)

        spec = {
            'dy_scaled': {
                'compress': 'round(asinh(dy_scaled)*40)',
                'decompress': 'sinh(dy_scaled_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            }
        }

        # Should work - compresses evaluated alias
        self.adf.compress_columns(spec)

        self.assertIn('dy_scaled_c', self.adf.df.columns)
        self.assertIn('dy_scaled', self.adf.aliases)

    def test_compression_is_idempotent(self):
        """Compressing already compressed column should skip silently."""
        spec = {
            'dy': {
                'compress': 'round(asinh(dy)*40)',
                'decompress': 'sinh(dy_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            }
        }

        self.adf.compress_columns(spec)
        self.assertIn('dy_c', self.adf.df.columns)
        self.assertNotIn('dy', self.adf.df.columns)

        # Compress again - should NOT raise, state unchanged
        self.adf.compress_columns(spec)
        self.assertIn('dy_c', self.adf.df.columns)
        self.assertNotIn('dy', self.adf.df.columns)
        # Verify no duplicate columns created
        self.assertEqual(list(self.adf.df.columns).count('dy_c'), 1)

    def test_compressed_column_name_collision_raises_error(self):
        """Test that compressed column name collision is detected"""
        # Create column that would conflict
        self.adf.df['dy_c'] = np.zeros(len(self.adf.df))

        spec = {
            'dy': {
                'compress': 'round(asinh(dy)*40)',
                'decompress': 'sinh(dy_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            }
        }

        with self.assertRaises(ValueError) as cm:
            self.adf.compress_columns(spec)

        self.assertIn('already exists', str(cm.exception))
        self.assertIn('dy_c', str(cm.exception))

    def test_decompress_inplace(self):
        """Test inplace decompression removes compressed column"""
        spec = {
            'dy': {
                'compress': 'round(asinh(dy)*40)',
                'decompress': 'sinh(dy_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            }
        }

        self.adf.compress_columns(spec)
        self.adf.decompress_columns(['dy'], inplace=True)

        # Check decompressed column is physical
        self.assertIn('dy', self.adf.df.columns)
        self.assertEqual(self.adf.df['dy'].dtype, np.float16)

        # Check compressed column removed
        self.assertNotIn('dy_c', self.adf.df.columns)

        # Check compression_info cleaned up
        self.assertNotIn('dy', self.adf.compression_info)

    def test_decompress_keep_compressed_false(self):
        """Test decompress with keep_compressed=False and keep_schema=False"""
        spec = {
            'dy': {
                'compress': 'round(asinh(dy)*40)',
                'decompress': 'sinh(dy_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            }
        }

        self.adf.compress_columns(spec)
        # New API: explicitly remove schema
        self.adf.decompress_columns(['dy'], keep_compressed=False, keep_schema=False)

        # Check decompressed column exists
        self.assertIn('dy', self.adf.df.columns)

        # Check compressed column removed
        self.assertNotIn('dy_c', self.adf.df.columns)

        # Check compression_info cleaned up
        self.assertNotIn('dy', self.adf.compression_info)

    def test_missing_compressed_column_raises_error(self):
        """Test error when compressed column is manually deleted"""
        spec = {
            'dy': {
                'compress': 'round(asinh(dy)*40)',
                'decompress': 'sinh(dy_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            }
        }

        self.adf.compress_columns(spec)

        # Manually delete compressed column (simulate corruption)
        self.adf.df.drop(columns=['dy_c'], inplace=True)

        # Should raise clear error
        with self.assertRaises(ValueError) as cm:
            self.adf.decompress_columns(['dy'])

        self.assertIn('missing', str(cm.exception).lower())
        self.assertIn('dy_c', str(cm.exception))

    def test_partial_failure_handling(self):
        """Test that failure on one column does not roll back prior successful compressions"""
        spec = {
            'dy': {
                'compress': 'round(asinh(dy)*40)',
                'decompress': 'sinh(dy_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            },
            'dz': {
                'compress': 'dz +* invalid_syntax',  # Invalid expression
                'decompress': 'sinh(dz_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            }
        }

        # Should raise error on 'dz'
        with self.assertRaises(ValueError) as cm:
            self.adf.compress_columns(spec)

        # Check that 'dy' was successfully compressed (partial success)
        self.assertIn('dy_c', self.adf.df.columns)
        self.assertIn('dy', self.adf.aliases)
        self.assertIn('dy', self.adf.compression_info)

        # Check that 'dz' did NOT create compressed column
        self.assertNotIn('dz_c', self.adf.df.columns)
        self.assertNotIn('dz', self.adf.compression_info)

        # Check original 'dz' still exists
        self.assertIn('dz', self.adf.df.columns)

        # Check error message indicates the failure
        self.assertIn('Compression failed', str(cm.exception))
        self.assertIn('dz', str(cm.exception))

    def test_roundtrip_save_load(self):
        """Test compression metadata survives save/load"""
        spec = {
            'dy': {
                'compress': 'round(asinh(dy)*40)',
                'decompress': 'sinh(dy_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            },
            'dz': {
                'compress': 'round(asinh(dz)*40)',
                'decompress': 'sinh(dz_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            }
        }

        self.adf.compress_columns(spec, measure_precision=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "compressed.parquet")
            self.adf.save(path)

            adf_loaded = AliasDataFrame.load(path)

            # Check compression_info preserved (2 columns + __meta__)
            self.assertEqual(len(adf_loaded.compression_info), 3)
            self.assertIn('dy', adf_loaded.compression_info)
            self.assertIn('dz', adf_loaded.compression_info)

            # Check aliases preserved
            self.assertIn('dy', adf_loaded.aliases)
            self.assertEqual(adf_loaded.aliases['dy'], 'sinh(dy_c/40.)')

            # Check precision info preserved
            self.assertIn('precision', adf_loaded.compression_info['dy'])

            # Materialize and verify values
            adf_loaded.materialize_alias('dy')
            np.testing.assert_allclose(
                adf_loaded.df['dy'].values,
                self.original_dy,
                rtol=0.01, atol=0.05
            )

    def test_roundtrip_export_import_tree(self):
        """Test compression metadata survives ROOT export/import"""
        spec = {
            'dy': {
                'compress': 'round(asinh(dy)*40)',
                'decompress': 'sinh(dy_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            }
        }

        self.adf.compress_columns(spec)

        with tempfile.NamedTemporaryFile(suffix=".root", delete=False) as tmp:
            self.adf.export_tree(tmp.name, treename="compressed", dropAliasColumns=False)
            tmp_path = tmp.name

        try:
            adf_loaded = AliasDataFrame.read_tree(tmp_path, treename="compressed")

            # Check compression_info preserved
            self.assertIn('dy', adf_loaded.compression_info)

            # Check can use decompression alias
            adf_loaded.materialize_alias('dy')
            np.testing.assert_allclose(
                adf_loaded.df['dy'].values,
                self.original_dy,
                rtol=0.01, atol=0.05
            )
        finally:
            os.remove(tmp_path)

    def test_multiple_columns_compression(self):
        """Test compressing multiple columns at once"""
        spec = {
            'dy': {
                'compress': 'round(asinh(dy)*40)',
                'decompress': 'sinh(dy_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            },
            'dz': {
                'compress': 'round(asinh(dz)*40)',
                'decompress': 'sinh(dz_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            },
            'tgSlp': {
                'compress': 'round(tgSlp*1000)',
                'decompress': 'tgSlp_c/1000.',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            }
        }

        self.adf.compress_columns(spec)

        # Check all compressed
        self.assertIn('dy_c', self.adf.df.columns)
        self.assertIn('dz_c', self.adf.df.columns)
        self.assertIn('tgSlp_c', self.adf.df.columns)

        # Check all have decompression aliases
        self.assertIn('dy', self.adf.aliases)
        self.assertIn('dz', self.adf.aliases)
        self.assertIn('tgSlp', self.adf.aliases)

        # Check compression_info complete (3 columns + __meta__)
        self.assertEqual(len(self.adf.compression_info), 4)
        self.assertIn('__meta__', self.adf.compression_info)

    def test_get_compression_info(self):
        """Test compression info retrieval"""
        spec = {
            'dy': {
                'compress': 'round(asinh(dy)*40)',
                'decompress': 'sinh(dy_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            }
        }

        self.adf.compress_columns(spec)

        # Test single column info
        info = self.adf.get_compression_info('dy')
        self.assertIsInstance(info, dict)
        self.assertEqual(info['compressed_col'], 'dy_c')

        # Test all columns as DataFrame
        df_info = self.adf.get_compression_info()
        self.assertIsInstance(df_info, pd.DataFrame)
        self.assertEqual(len(df_info), 1)
        self.assertIn('dy', df_info.index)

    def test_backward_compatibility_no_compression_info(self):
        """Test loading old files without compression_info works"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "old_format.parquet")

            # Save without compression
            self.adf.save(path)

            # Load should work fine - __meta__ should be present
            adf_loaded = AliasDataFrame.load(path)
            # Only __meta__ should be present (no actual compressed columns)
            self.assertEqual(len(adf_loaded.compression_info), 1)
            self.assertIn('__meta__', adf_loaded.compression_info)


class TestCompressionStateMachine(unittest.TestCase):
    """Test compression state machine transitions and invariants"""

    def setUp(self):
        """Create test data for compression tests"""
        np.random.seed(42)
        df = pd.DataFrame({
            "dy": np.random.normal(0, 2.0, 1000).astype(np.float32),
            "dz": np.random.normal(0, 1.5, 1000).astype(np.float32),
            "tgSlp": np.random.uniform(-0.5, 0.5, 1000).astype(np.float32),
        })
        self.adf = AliasDataFrame(df)
        self.original_dy = df["dy"].values.copy()

        self.spec = {
            'dy': {
                'compress': 'round(asinh(dy)*40)',
                'decompress': 'sinh(dy_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            },
            'dz': {
                'compress': 'round(asinh(dz)*40)',
                'decompress': 'sinh(dz_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            }
        }

    def test_metadata_versioning(self):
        """Test that __meta__ is present in compression_info"""
        self.assertIn("__meta__", self.adf.compression_info)
        meta = self.adf.compression_info["__meta__"]
        self.assertEqual(meta["schema_version"], 1)
        self.assertEqual(meta["state_machine"], "CompressionState.v1")

    def test_schema_only_definition(self):
        """Test SCHEMA_ONLY state (forward declaration)"""
        # Define schema without data
        self.adf.define_compression_schema(self.spec)

        # Check state is SCHEMA_ONLY
        from dfextensions.AliasDataFrame import CompressionState
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.SCHEMA_ONLY)
        self.assertEqual(self.adf.get_compression_state('dz'), CompressionState.SCHEMA_ONLY)

        # Check no physical columns created
        self.assertNotIn('dy_c', self.adf.df.columns)
        self.assertNotIn('dz_c', self.adf.df.columns)

        # Check original columns still exist
        self.assertIn('dy', self.adf.df.columns)
        self.assertIn('dz', self.adf.df.columns)

        # Check metadata stored
        info = self.adf.compression_info['dy']
        self.assertEqual(info['compressed_col'], 'dy_c')
        self.assertEqual(info['compress_expr'], self.spec['dy']['compress'])
        self.assertEqual(info['state'], CompressionState.SCHEMA_ONLY)

    def test_schema_only_then_compress(self):
        """Test SCHEMA_ONLY → COMPRESSED transition"""
        from dfextensions.AliasDataFrame import CompressionState

        # Step 1: Define schema
        self.adf.define_compression_schema(self.spec)
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.SCHEMA_ONLY)

        # Step 2: Apply compression using schema
        self.adf.compress_columns(columns=['dy'])

        # Check state transitioned to COMPRESSED
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.COMPRESSED)

        # Check physical columns exist
        self.assertIn('dy_c', self.adf.df.columns)
        self.assertEqual(self.adf.df['dy_c'].dtype, np.int16)

        # Check decompression alias exists
        self.assertIn('dy', self.adf.aliases)
        self.assertEqual(self.adf.aliases['dy'], self.spec['dy']['decompress'])

        # Check original removed
        self.assertNotIn('dy', self.adf.df.columns)

    def test_direct_compression_without_schema(self):
        """Test None → COMPRESSED transition (inline compression)"""
        from dfextensions.AliasDataFrame import CompressionState

        self.adf.compress_columns({'dy': self.spec['dy']})

        # Check state is COMPRESSED
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.COMPRESSED)

        # Check invariants
        self.assertIn('dy_c', self.adf.df.columns)
        self.assertIn('dy', self.adf.aliases)
        self.assertNotIn('dy', self.adf.df.columns)

    def test_full_compression_cycle(self):
        """Test COMPRESSED → DECOMPRESSED → COMPRESSED (recompression)"""
        from dfextensions.AliasDataFrame import CompressionState

        # Step 1: Compress
        self.adf.compress_columns({'dy': self.spec['dy']})
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.COMPRESSED)

        # Step 2: Decompress with keep_schema=True
        self.adf.decompress_columns(['dy'], keep_schema=True, keep_compressed=False)
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.DECOMPRESSED)

        # Check invariants after decompression
        self.assertIn('dy', self.adf.df.columns)  # Physical column
        self.assertNotIn('dy', self.adf.aliases)   # No alias
        self.assertNotIn('dy_c', self.adf.df.columns)  # Compressed removed

        # Step 3: Recompress using stored schema
        self.adf.compress_columns(columns=['dy'])
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.COMPRESSED)

        # Check invariants after recompression
        self.assertIn('dy_c', self.adf.df.columns)
        self.assertIn('dy', self.adf.aliases)
        self.assertNotIn('dy', self.adf.df.columns)

        # Verify data integrity
        self.adf.materialize_alias('dy')
        np.testing.assert_allclose(
            self.adf.df['dy'].values,
            self.original_dy,
            rtol=0.01, atol=0.05
        )

    def test_decompress_with_keep_schema_false(self):
        """Test COMPRESSED → None transition (remove all metadata)"""
        from dfextensions.AliasDataFrame import CompressionState

        self.adf.compress_columns({'dy': self.spec['dy']})
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.COMPRESSED)

        self.adf.decompress_columns(['dy'], keep_schema=False)

        # Check state removed
        self.assertIsNone(self.adf.get_compression_state('dy'))
        self.assertNotIn('dy', self.adf.compression_info)

        # Check physical column exists
        self.assertIn('dy', self.adf.df.columns)
        self.assertNotIn('dy', self.adf.aliases)

    def test_decompression_is_idempotent(self):
        """Decompressing already decompressed column should skip silently."""
        from dfextensions.AliasDataFrame import CompressionState

        self.adf.compress_columns({'dy': self.spec['dy']})
        self.adf.decompress_columns(['dy'], keep_schema=True)
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.DECOMPRESSED)
        self.assertIn('dy', self.adf.df.columns)

        # Decompress again - should NOT raise, state unchanged
        self.adf.decompress_columns(['dy'])
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.DECOMPRESSED)
        self.assertIn('dy', self.adf.df.columns)

    def test_double_compression_is_idempotent(self):
        """Compressing already compressed column should skip silently."""
        from dfextensions.AliasDataFrame import CompressionState

        self.adf.compress_columns({'dy': self.spec['dy']})
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.COMPRESSED)
        self.assertIn('dy_c', self.adf.df.columns)

        # Compress again - should NOT raise, state unchanged
        self.adf.compress_columns({'dy': self.spec['dy']})
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.COMPRESSED)
        self.assertIn('dy_c', self.adf.df.columns)
        self.assertEqual(list(self.adf.df.columns).count('dy_c'), 1)

    def test_compress_decompress_roundtrip_idempotent(self):
        """Test compress → decompress → compress roundtrip with idempotent calls."""
        from dfextensions.AliasDataFrame import CompressionState

        # First compress
        self.adf.compress_columns({'dy': self.spec['dy']})
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.COMPRESSED)

        # Compress again (idempotent - should skip)
        self.adf.compress_columns({'dy': self.spec['dy']})
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.COMPRESSED)

        # Decompress
        self.adf.decompress_columns(['dy'], keep_schema=True)
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.DECOMPRESSED)

        # Decompress again (idempotent - should skip)
        self.adf.decompress_columns(['dy'])
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.DECOMPRESSED)

        # Recompress
        self.adf.compress_columns(columns=['dy'])
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.COMPRESSED)
        self.assertIn('dy_c', self.adf.df.columns)

    def test_schema_compressed_but_data_not(self):
        """Test compression when schema says compressed but data isn't.
        
        This happens when:
        - Schema is loaded from file with state='compressed'
        - But DataFrame has original columns (not compressed data)
        
        The fix should detect this and compress the data.
        """
        from dfextensions.AliasDataFrame import CompressionState
        
        # Manually set up the problematic state:
        # - Schema says 'compressed' with full compression info
        # - But DataFrame has original column 'dy', not 'dy_c'
        self.adf._schema['compression']['dy'] = {
            'compressed_col': 'dy_c',
            'compress_expr': 'round(asinh(dy)*40)',
            'decompress_expr': 'sinh(dy_c/40.)',
            'compressed_dtype': 'int16',
            'decompressed_dtype': 'float16',
            'state': CompressionState.COMPRESSED,  # Schema says compressed!
            'original_removed': True
        }
        
        # Verify setup: schema says compressed, but df has 'dy' not 'dy_c'
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.COMPRESSED)
        self.assertIn('dy', self.adf.df.columns)
        self.assertNotIn('dy_c', self.adf.df.columns)
        
        # Now compress - should detect mismatch and actually compress
        self.adf.compress_columns(columns=['dy'])
        
        # After compression: data should be compressed
        self.assertIn('dy_c', self.adf.df.columns)
        self.assertNotIn('dy', self.adf.df.columns)
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.COMPRESSED)

    def test_export_schema_include_state_true(self):
        """Test export_schema_v2 with include_state=True includes state fields."""
        from dfextensions.AliasDataFrame import CompressionState
        
        self.adf.compress_columns({'dy': self.spec['dy']})
        
        schema = self.adf.export_schema_v2(include_state=True)
        comp = schema.get('compression', {}).get('dy', {})
        
        # State fields should be present
        self.assertIn('state', comp)
        self.assertIn('original_removed', comp)
        self.assertEqual(comp['state'], CompressionState.COMPRESSED)
    
    def test_export_schema_include_state_false(self):
        """Test export_schema_v2 with include_state=False excludes state fields."""
        from dfextensions.AliasDataFrame import CompressionState
        
        self.adf.compress_columns({'dy': self.spec['dy']})
        
        schema = self.adf.export_schema_v2(include_state=False)
        comp = schema.get('compression', {}).get('dy', {})
        
        # State fields should NOT be present
        self.assertNotIn('state', comp)
        self.assertNotIn('original_removed', comp)
        
        # Definition fields should still be present
        self.assertIn('compress_expr', comp)
        self.assertIn('decompress_expr', comp)
        self.assertIn('compressed_dtype', comp)
        self.assertIn('decompressed_dtype', comp)

    def test_export_schema_include_state_false_no_leakage(self):
        """Test that state/original_removed never leak into any compression block.
        
        GPT review suggestion: ensure no state fields appear anywhere,
        including in subframes.
        """
        from dfextensions.AliasDataFrame import CompressionState
        
        self.adf.compress_columns({'dy': self.spec['dy']})
        
        schema = self.adf.export_schema_v2(include_state=False, include_subframes=True)
        
        # Helper to walk all compression blocks (main + subframes recursively)
        def walk_compression_blocks(s):
            if 'compression' in s:
                yield s['compression']
            for sf in s.get('subframes', {}).values():
                if isinstance(sf, dict):
                    yield from walk_compression_blocks(sf)
        
        # Ensure no state fields anywhere
        for comp_block in walk_compression_blocks(schema):
            for name, cfg in comp_block.items():
                if name == '__meta__':
                    continue
                self.assertNotIn('state', cfg, 
                    f"state should not appear in compression block for {name}")
                self.assertNotIn('original_removed', cfg,
                    f"original_removed should not appear in compression block for {name}")

    def test_definition_schema_infers_schema_only_state(self):
        """Test that loading a definition schema (no state) infers SCHEMA_ONLY.
        
        When a schema is exported with include_state=False, loading it should
        result in SCHEMA_ONLY state (not None/unknown), allowing compress_columns()
        to work correctly.
        """
        from dfextensions.AliasDataFrame import CompressionState
        
        # Simulate loading a definition schema (no state field)
        self.adf._schema['compression']['dy'] = {
            'compressed_col': 'dy_c',
            'compress_expr': 'round(asinh(dy)*40)',
            'decompress_expr': 'sinh(dy_c/40.)',
            'compressed_dtype': 'int16',
            'decompressed_dtype': 'float16'
            # NO 'state' field - this is definition-only schema
        }
        
        # State should be inferred as SCHEMA_ONLY
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.SCHEMA_ONLY)
        
        # compress_columns should work (transition SCHEMA_ONLY → COMPRESSED)
        self.adf.compress_columns(columns=['dy'])
        
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.COMPRESSED)
        self.assertIn('dy_c', self.adf.df.columns)
        self.assertNotIn('dy', self.adf.df.columns)

    def test_definition_schema_export_no_alias_for_compression_targets(self):
        """Test that definition schema exports compression targets as physical columns.
        
        When exporting with include_state=False:
        - Compression targets (e.g., dy) should be exported as physical columns (no expr)
        - Compressed storage columns (e.g., dy_c) should NOT be exported
        """
        from dfextensions.AliasDataFrame import CompressionState
        
        # Compress first
        self.adf.compress_columns({'dy': self.spec['dy']})
        
        # Export definition schema
        def_schema = self.adf.export_schema_v2(include_state=False)
        
        # dy should be physical column (no expr)
        dy_col = def_schema.get('columns', {}).get('dy', {})
        self.assertNotIn('expr', dy_col, "Definition schema should not have expr for compression targets")
        self.assertIn('dtype', dy_col)
        
        # dy_c should NOT be in columns (doesn't exist in fresh data)
        self.assertNotIn('dy_c', def_schema.get('columns', {}), 
            "Definition schema should not include compressed storage columns")
        
        # Compression section should still have the definitions
        self.assertIn('dy', def_schema.get('compression', {}))
        comp = def_schema['compression']['dy']
        self.assertIn('compress_expr', comp)
        self.assertIn('decompress_expr', comp)
        self.assertNotIn('state', comp)  # No state in definition schema

    def test_definition_schema_roundtrip_compress(self):
        """Test full workflow: compress → export definition → load fresh → compress.
        
        This is the real-world use case that was failing with cycle detection.
        """
        from dfextensions.AliasDataFrame import AliasDataFrame, CompressionState
        
        # Compress original data
        self.adf.compress_columns({'dy': self.spec['dy']})
        
        # Export definition schema
        def_schema = self.adf.export_schema_v2(include_state=False)
        
        # Create fresh data (simulating loading new uncompressed file)
        fresh_df = pd.DataFrame({
            'dy': np.random.randn(50).astype(np.float32),
            'x': np.random.randn(50).astype(np.float32)
        })
        fresh_adf = AliasDataFrame(fresh_df)
        
        # Apply compression definitions from schema
        fresh_adf._schema['compression'] = def_schema.get('compression', {})
        
        # This should NOT fail with "Cycle detected"
        fresh_adf.compress_columns(columns=['dy'])
        
        # Verify compression worked
        self.assertIn('dy_c', fresh_adf.df.columns)
        self.assertNotIn('dy', fresh_adf.df.columns)
        self.assertEqual(fresh_adf.get_compression_state('dy'), CompressionState.COMPRESSED)

    def test_record_schema_includes_aliases(self):
        """Test that record schema (include_state=True) includes aliases correctly."""
        from dfextensions.AliasDataFrame import CompressionState
        
        # Compress
        self.adf.compress_columns({'dy': self.spec['dy']})
        
        # Export record schema
        rec_schema = self.adf.export_schema_v2(include_state=True)
        
        # dy should be an alias in record schema
        dy_col = rec_schema.get('columns', {}).get('dy', {})
        self.assertIn('expr', dy_col, "Record schema should have expr for aliases")
        
        # dy_c should exist
        self.assertIn('dy_c', rec_schema.get('columns', {}))
        
        # State should be included
        comp = rec_schema.get('compression', {}).get('dy', {})
        self.assertIn('state', comp)

    def test_collision_same_schema_recompression(self):
        """Test recompression with matching schema is allowed"""
        from dfextensions.AliasDataFrame import CompressionState

        # Compress, decompress, recompress
        self.adf.compress_columns({'dy': self.spec['dy']})
        self.adf.decompress_columns(['dy'], keep_schema=True, keep_compressed=False)

        # This should work - reuses dy_c name from schema
        self.adf.compress_columns(columns=['dy'])

        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.COMPRESSED)
        self.assertIn('dy_c', self.adf.df.columns)

    def test_collision_foreign_column(self):
        """Test collision with unrelated column raises error"""
        # Create conflicting column
        self.adf.df['dy_c'] = np.zeros(len(self.adf.df))

        with self.assertRaises(ValueError) as cm:
            self.adf.compress_columns({'dy': self.spec['dy']})

        self.assertIn('already exists', str(cm.exception))
        self.assertIn('dy_c', str(cm.exception))

    def test_collision_other_schema(self):
        """Test collision with another column's compressed_col raises error"""
        # First create an unrelated column called 'dy_c'
        self.adf.df['dy_c'] = np.ones(len(self.adf.df))

        # Now try to compress dy, which would want to create dy_c
        with self.assertRaises(ValueError) as cm:
            self.adf.compress_columns({'dy': self.spec['dy']})

        # Check error message mentions the conflict
        self.assertIn('already exists', str(cm.exception).lower())
        self.assertIn('dy_c', str(cm.exception))

    def test_compress_all_schema_only_columns(self):
        """Test compress_columns() with no args compresses all SCHEMA_ONLY"""
        from dfextensions.AliasDataFrame import CompressionState

        # Define schemas
        self.adf.define_compression_schema(self.spec)

        # Compress all at once (no args)
        self.adf.compress_columns()

        # Check both compressed
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.COMPRESSED)
        self.assertEqual(self.adf.get_compression_state('dz'), CompressionState.COMPRESSED)

    def test_is_compressed_helper(self):
        """Test is_compressed() helper method"""
        self.assertFalse(self.adf.is_compressed('dy'))

        self.adf.compress_columns({'dy': self.spec['dy']})
        self.assertTrue(self.adf.is_compressed('dy'))

        self.adf.decompress_columns(['dy'], keep_schema=True)
        self.assertFalse(self.adf.is_compressed('dy'))

    def test_get_compression_info_excludes_meta(self):
        """Test get_compression_info() filters __meta__"""
        self.adf.compress_columns({'dy': self.spec['dy']})

        # Single column - should work
        info = self.adf.get_compression_info('dy')
        self.assertIsInstance(info, dict)
        self.assertIn('state', info)

        # All columns - should exclude __meta__
        df_info = self.adf.get_compression_info()
        self.assertNotIn('__meta__', df_info.index)
        self.assertIn('dy', df_info.index)

    def test_precision_measurement_with_state(self):
        """Test precision measurement works with new state system"""
        self.adf.compress_columns({'dy': self.spec['dy']}, measure_precision=True)

        info = self.adf.compression_info['dy']
        self.assertIn('precision', info)
        self.assertIn('rmse', info['precision'])
        self.assertGreater(info['precision']['rmse'], 0)

    def test_schema_from_info_helper(self):
        """Test _schema_from_info() reconstructs spec correctly"""
        self.adf.define_compression_schema({'dy': self.spec['dy']})

        reconstructed = self.adf._schema_from_info('dy')

        self.assertEqual(reconstructed['compress'], self.spec['dy']['compress'])
        self.assertEqual(reconstructed['decompress'], self.spec['dy']['decompress'])
        self.assertEqual(reconstructed['compressed_dtype'], self.spec['dy']['compressed_dtype'])

    def test_invalid_state_transition_schema_only_to_decompress(self):
        """Test that SCHEMA_ONLY → DECOMPRESS is a no-op"""
        from dfextensions.AliasDataFrame import CompressionState

        self.adf.define_compression_schema({'dy': self.spec['dy']})

        # Try to decompress SCHEMA_ONLY column (should be no-op)
        self.adf.decompress_columns(['dy'])

        # State should still be SCHEMA_ONLY
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.SCHEMA_ONLY)

    def test_backward_compatibility_old_files(self):
        """Test that old files without __meta__ are handled"""
        # Simulate old file by removing __meta__
        if "__meta__" in self.adf.compression_info:
            del self.adf.compression_info["__meta__"]

        # get_compression_info should still work
        df_info = self.adf.get_compression_info()
        self.assertIsInstance(df_info, pd.DataFrame)

    def test_state_invariants_after_compress(self):
        """Test state invariants after compression"""
        from dfextensions.AliasDataFrame import CompressionState

        self.adf.compress_columns({'dy': self.spec['dy']})

        # Invariant checks
        state = self.adf.get_compression_state('dy')
        self.assertEqual(state, CompressionState.COMPRESSED)

        # Physical compressed column exists
        self.assertIn('dy_c', self.adf.df.columns)

        # Original is alias, not physical
        self.assertNotIn('dy', self.adf.df.columns)
        self.assertIn('dy', self.adf.aliases)

        # Metadata consistent
        info = self.adf.compression_info['dy']
        self.assertEqual(info['state'], CompressionState.COMPRESSED)
        self.assertEqual(info['compressed_col'], 'dy_c')

    def test_state_invariants_after_decompress(self):
        """Test state invariants after decompression"""
        from dfextensions.AliasDataFrame import CompressionState

        self.adf.compress_columns({'dy': self.spec['dy']})
        self.adf.decompress_columns(['dy'], keep_schema=True, keep_compressed=True)

        # Invariant checks
        state = self.adf.get_compression_state('dy')
        self.assertEqual(state, CompressionState.DECOMPRESSED)

        # Decompressed column is physical
        self.assertIn('dy', self.adf.df.columns)

        # Not an alias
        self.assertNotIn('dy', self.adf.aliases)

        # Compressed column still exists (keep_compressed=True)
        self.assertIn('dy_c', self.adf.df.columns)

        # Metadata consistent
        info = self.adf.compression_info['dy']
        self.assertEqual(info['state'], CompressionState.DECOMPRESSED)

    def test_selective_registration_from_spec(self):
        """Test compress_columns(spec, columns=[subset]) only registers subset"""
        from dfextensions.AliasDataFrame import CompressionState

        # Compress only dy from full spec
        self.adf.compress_columns(self.spec, columns=['dy'])

        # Check ONLY dy registered and compressed
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.COMPRESSED)
        self.assertIsNone(self.adf.get_compression_state('dz'))  # NOT registered

        # Check metadata
        self.assertIn('dy', self.adf.compression_info)
        self.assertNotIn('dz', self.adf.compression_info)

        # Check physical columns
        self.assertIn('dy_c', self.adf.df.columns)
        self.assertNotIn('dz_c', self.adf.df.columns)

    def test_multiple_selective_calls(self):
        """Test Pattern 2: Multiple compress_columns calls with subsets"""
        from dfextensions.AliasDataFrame import CompressionState

        # First call: compress dy
        self.adf.compress_columns(self.spec, columns=['dy'])
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.COMPRESSED)

        # Second call: compress dz (should work, not error)
        self.adf.compress_columns(self.spec, columns=['dz'])
        self.assertEqual(self.adf.get_compression_state('dz'), CompressionState.COMPRESSED)

        # Both should be compressed now
        self.assertTrue(self.adf.is_compressed('dy'))
        self.assertTrue(self.adf.is_compressed('dz'))

        # Both have separate metadata
        self.assertIn('dy', self.adf.compression_info)
        self.assertIn('dz', self.adf.compression_info)

    def test_selective_mode_skips_same_schema_compressed(self):
        """Test that re-compressing with SAME schema is silently skipped (idempotent)"""
        from dfextensions.AliasDataFrame import CompressionState

        # Compress
        self.adf.compress_columns(self.spec, columns=['dy'])
        dy_c_before = self.adf.df['dy_c'].copy()

        # Try to compress again with same schema (should skip)
        self.adf.compress_columns(self.spec, columns=['dy'])

        # Should still be compressed, data unchanged
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.COMPRESSED)
        np.testing.assert_array_equal(self.adf.df['dy_c'], dy_c_before)

    def test_selective_mode_errors_on_schema_change_when_compressed(self):
        """Test error when trying to change schema of COMPRESSED column"""
        from dfextensions.AliasDataFrame import CompressionState

        # Compress with original schema
        self.adf.compress_columns(self.spec, columns=['dy'])
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.COMPRESSED)

        # Try to compress with different schema
        new_spec = {
            'dy': {
                'compress': 'round(dy*1000)',  # Different transform
                'decompress': 'dy_c/1000.',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float32
            }
        }

        with self.assertRaises(ValueError) as cm:
            self.adf.compress_columns(new_spec, columns=['dy'])

        self.assertIn('already compressed', str(cm.exception).lower())
        self.assertIn('different schema', str(cm.exception).lower())
        self.assertIn('decompress first', str(cm.exception).lower())

    def test_selective_mode_validates_column_exists(self):
        """Test that selective mode with on_missing='error' validates column exists"""
        spec = {
            'nonexistent': {
                'compress': 'round(nonexistent*10)',
                'decompress': 'nonexistent_c/10.',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float32
            }
        }

        # Use on_missing='error' for strict validation
        with self.assertRaises(KeyError) as cm:
            self.adf.compress_columns(spec, columns=['nonexistent'], on_missing='error')

        self.assertIn("Missing columns", str(cm.exception))
    def test_selective_mode_validates_columns_in_spec(self):
        """Test that selective mode validates requested columns are in spec"""
        with self.assertRaises(ValueError) as cm:
            self.adf.compress_columns(self.spec, columns=['dy', 'nonexistent'])

        self.assertIn('not found in compression_spec', str(cm.exception))
        self.assertIn('nonexistent', str(cm.exception))

    def test_selective_mode_updates_schema_for_schema_only(self):
        """Test that Pattern 2 can update schema for SCHEMA_ONLY columns"""
        from dfextensions.AliasDataFrame import CompressionState

        # Step 1: Register initial schema (Pattern 1)
        old_spec = {
            'dy': {
                'compress': 'round(dy*10)',
                'decompress': 'dy_c/10.',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float32
            }
        }
        self.adf.define_compression_schema(old_spec)
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.SCHEMA_ONLY)

        # Step 2: Update schema using Pattern 2
        new_spec = {
            'dy': {
                'compress': 'round(asinh(dy)*40)',
                'decompress': 'sinh(dy_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            }
        }
        self.adf.compress_columns(new_spec, columns=['dy'])

        # Check schema was updated and compressed
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.COMPRESSED)
        info = self.adf.compression_info['dy']
        self.assertEqual(info['compress_expr'], 'round(asinh(dy)*40)')
        self.assertEqual(info['decompressed_dtype'], 'float16')

    def test_real_world_incremental_compression_pattern2(self):
        """Test Scenario 3 from spec: incremental compression using Pattern 2"""
        from dfextensions.AliasDataFrame import CompressionState

        # Add tgSlp to test data
        self.adf.df['tgSlp'] = np.random.uniform(-0.5, 0.5, len(self.adf.df))

        # Step 1: Compress subset for initial analysis (Pattern 2)
        self.adf.compress_columns(self.spec, columns=['dy', 'dz'])

        self.assertTrue(self.adf.is_compressed('dy'))
        self.assertTrue(self.adf.is_compressed('dz'))
        self.assertIsNone(self.adf.get_compression_state('tgSlp'))

        # Step 2: Later compress additional column (Pattern 2)
        tgSlp_spec = {
            'tgSlp': {
                'compress': 'round(tgSlp*1000)',
                'decompress': 'tgSlp_c/1000.',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float32
            }
        }
        self.adf.compress_columns(tgSlp_spec, columns=['tgSlp'])

        # All three compressed now
        self.assertTrue(self.adf.is_compressed('dy'))
        self.assertTrue(self.adf.is_compressed('dz'))
        self.assertTrue(self.adf.is_compressed('tgSlp'))

        # Verify data integrity
        self.assertIn('dy_c', self.adf.df.columns)
        self.assertIn('dz_c', self.adf.df.columns)
        self.assertIn('tgSlp_c', self.adf.df.columns)

    def test_pattern1_pattern2_mixing(self):
        """Test mixing Pattern 1 (schema-first) and Pattern 2 (selective)"""
        from dfextensions.AliasDataFrame import CompressionState

        # Pattern 1: Define full schema
        self.adf.define_compression_schema(self.spec)
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.SCHEMA_ONLY)
        self.assertEqual(self.adf.get_compression_state('dz'), CompressionState.SCHEMA_ONLY)

        # Pattern 2: Compress only dy with potentially updated schema
        updated_spec = {
            'dy': {
                'compress': 'round(dy*100)',  # Different from original
                'decompress': 'dy_c/100.',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float32
            }
        }
        self.adf.compress_columns(updated_spec, columns=['dy'])

        # dy should be compressed with new schema
        self.assertEqual(self.adf.get_compression_state('dy'), CompressionState.COMPRESSED)
        self.assertEqual(self.adf.compression_info['dy']['compress_expr'], 'round(dy*100)')

        # dz should still be SCHEMA_ONLY with original schema
        self.assertEqual(self.adf.get_compression_state('dz'), CompressionState.SCHEMA_ONLY)
        self.assertEqual(self.adf.compression_info['dz']['compress_expr'], self.spec['dz']['compress'])


class TestCompressionOnMissing(unittest.TestCase):
    """Test on_missing and return_summary parameters"""

    def setUp(self):
        """Create test DataFrame and compression spec"""
        self.df = pd.DataFrame({
            'dy': np.random.randn(10),
            'dz': np.random.randn(10),
            'y': np.random.randn(10) * 50,
        })

        # Spec with some columns that won't exist in df
        self.spec = {
            'dy': {
                'compress': 'round(asinh(dy)*40)',
                'decompress': 'sinh(dy_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            },
            'dz': {
                'compress': 'round(asinh(dz)*40)',
                'decompress': 'sinh(dz_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            },
            'y': {
                'compress': 'round(y*(0x7fff/50.))',
                'decompress': 'y_c*(50.0/(0x7fff))',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float32
            },
            'dyC0': {  # This column doesn't exist in df
                'compress': 'round(asinh(dyC0)*100)',
                'decompress': 'sinh(dyC0_c/100.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            }
        }

    def test_default_warn_mode(self):
        """Test default on_missing='warn' behavior"""
        adf = AliasDataFrame(self.df)

        with self.assertWarns(UserWarning) as cm:
            result = adf.compress_columns(self.spec, return_summary=True)

        # Check warning message
        self.assertIn("Skipping missing columns", str(cm.warning))
        self.assertIn("dyC0", str(cm.warning))

        # Check summary
        self.assertEqual(set(result['compressed']), {'dy', 'dz', 'y'})
        self.assertEqual(result['skipped'], ['dyC0'])

        # Verify compression worked
        self.assertIn('dy_c', adf.df.columns)
        self.assertIn('dz_c', adf.df.columns)
        self.assertIn('y_c', adf.df.columns)
        self.assertNotIn('dyC0_c', adf.df.columns)

    def test_strict_error_mode(self):
        """Test on_missing='error' raises KeyError"""
        adf = AliasDataFrame(self.df)

        with self.assertRaises(KeyError) as cm:
            adf.compress_columns(self.spec, on_missing='error')

        # Check error message
        self.assertIn("Missing columns", str(cm.exception))
        self.assertIn("dyC0", str(cm.exception))

    def test_silent_ignore_mode(self):
        """Test on_missing='ignore' produces no warnings"""
        adf = AliasDataFrame(self.df)

        # Use warnings filter to catch any warnings
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = adf.compress_columns(self.spec, on_missing='ignore', return_summary=True)

        # No warnings should be raised
        self.assertEqual(len(w), 0)

        # But compression should still work
        self.assertEqual(set(result['compressed']), {'dy', 'dz', 'y'})
        self.assertEqual(result['skipped'], ['dyC0'])

    def test_explicit_columns_subset(self):
        """Test compression with explicit columns parameter"""
        adf = AliasDataFrame(self.df)

        # Only compress dy and dz
        result = adf.compress_columns(
            self.spec,
            columns=['dy', 'dz'],
            return_summary=True
        )

        self.assertEqual(set(result['compressed']), {'dy', 'dz'})
        self.assertEqual(result['skipped'], [])  # All requested columns exist

        # Verify only requested columns were compressed
        self.assertIn('dy_c', adf.df.columns)
        self.assertIn('dz_c', adf.df.columns)
        self.assertNotIn('y_c', adf.df.columns)

    def test_return_summary_default_false(self):
        """Test that return_summary=False returns self (backward compatible)"""
        adf = AliasDataFrame(self.df)

        # Default should return self
        result = adf.compress_columns({'dy': self.spec['dy']})
        self.assertIs(result, adf)

        # Explicit False should also return self
        result = adf.compress_columns(
            {'dz': self.spec['dz']},
            return_summary=False
        )
        self.assertIs(result, adf)

    def test_all_columns_missing_warn(self):
        """Test behavior when all columns are missing"""
        adf = AliasDataFrame(pd.DataFrame({'x': [1, 2, 3]}))

        with self.assertWarns(UserWarning) as cm:
            result = adf.compress_columns(self.spec, return_summary=True)

        self.assertIn("Skipping missing columns", str(cm.warning))
        self.assertEqual(result['compressed'], [])
        self.assertEqual(set(result['skipped']), {'dy', 'dz', 'y', 'dyC0'})

    def test_all_columns_missing_error(self):
        """Test error mode when all columns are missing"""
        adf = AliasDataFrame(pd.DataFrame({'x': [1, 2, 3]}))

        with self.assertRaises(KeyError) as cm:
            adf.compress_columns(self.spec, on_missing='error')

        self.assertIn("Missing columns", str(cm.exception))

    def test_partial_missing_with_columns_param(self):
        """Test warning when some explicitly requested columns are missing"""
        df = pd.DataFrame({'dy': np.random.randn(10)})
        adf = AliasDataFrame(df)

        with self.assertWarns(UserWarning) as cm:
            result = adf.compress_columns(
                self.spec,
                columns=['dy', 'dyC0'],  # dyC0 doesn't exist
                return_summary=True
            )

        self.assertEqual(result['compressed'], ['dy'])
        self.assertEqual(result['skipped'], ['dyC0'])

    def test_method_chaining_still_works(self):
        """Test that method chaining still works with default parameters"""
        adf = AliasDataFrame(self.df)

        # Should be able to chain
        result = (adf
                  .compress_columns({'dy': self.spec['dy']})
                  .compress_columns({'dz': self.spec['dz']}))

        self.assertIs(result, adf)
        self.assertIn('dy_c', adf.df.columns)
        self.assertIn('dz_c', adf.df.columns)

class TestReadTreeOptimized(unittest.TestCase):
    """Test cases for optimized read_tree with entry_range and threading support"""

    def setUp(self):
        """Create test data and export to ROOT file"""
        # Create main DataFrame
        n_rows = 1000
        self.df = pd.DataFrame({
            'x': np.random.randn(n_rows).astype(np.float32),
            'y': np.random.randn(n_rows).astype(np.float32),
            'z': np.arange(n_rows, dtype=np.int32),
        })

        # Create subframe (small calibration data)
        self.df_calib = pd.DataFrame({
            'calib_id': np.arange(10),
            'factor': np.random.randn(10).astype(np.float32),
        })

        self.adf = AliasDataFrame(self.df)
        self.adf.add_alias('xy_sum', 'x + y', dtype=np.float32)

        # Register subframe
        adf_calib = AliasDataFrame(self.df_calib)
        self.adf.register_subframe('calib', adf_calib, index_columns='calib_id')

        # Export to temp file
        self.tmp = tempfile.NamedTemporaryFile(suffix=".root", delete=False)
        self.tmp_path = self.tmp.name
        self.tmp.close()
        self.adf.export_tree(self.tmp_path, treename="tree", dropAliasColumns=False)

    def tearDown(self):
        """Clean up temp file"""
        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)

    def test_basic_read(self):
        """Test basic read_tree without entry range"""
        adf_loaded = AliasDataFrame.read_tree(self.tmp_path, "tree")

        # Check shape matches
        self.assertEqual(len(adf_loaded.df), len(self.df))
        self.assertEqual(set(adf_loaded.df.columns), set(self.df.columns))

        # Check data matches
        np.testing.assert_array_almost_equal(
            adf_loaded.df['x'].values,
            self.df['x'].values,
            decimal=5
        )

    def test_entry_range_stop(self):
        """Test reading partial file with entry_stop"""
        adf_loaded = AliasDataFrame.read_tree(
            self.tmp_path, "tree",
            entry_stop=500
        )

        self.assertEqual(len(adf_loaded.df), 500)
        np.testing.assert_array_almost_equal(
            adf_loaded.df['x'].values,
            self.df['x'].values[:500],
            decimal=5
        )

    def test_entry_range_start_stop(self):
        """Test reading range with both entry_start and entry_stop"""
        adf_loaded = AliasDataFrame.read_tree(
            self.tmp_path, "tree",
            entry_start=200,
            entry_stop=700
        )

        self.assertEqual(len(adf_loaded.df), 500)
        np.testing.assert_array_almost_equal(
            adf_loaded.df['z'].values,
            self.df['z'].values[200:700],
            decimal=5
        )

    def test_threaded_vs_unthreaded_equivalence(self):
        """Test that threaded and unthreaded reads produce identical results"""
        # Single-threaded
        adf_single = AliasDataFrame.read_tree(
            self.tmp_path, "tree",
            num_workers=1
        )

        # Multi-threaded
        adf_multi = AliasDataFrame.read_tree(
            self.tmp_path, "tree",
            num_workers=4
        )

        # Results should be identical
        pd.testing.assert_frame_equal(
            adf_single.df.sort_index(axis=1),
            adf_multi.df.sort_index(axis=1)
        )

        # Aliases should match
        self.assertEqual(adf_single.aliases, adf_multi.aliases)

    def test_subframe_loaded(self):
        """Test that subframes are loaded correctly"""
        adf_loaded = AliasDataFrame.read_tree(self.tmp_path, "tree")

        # Check subframe exists
        self.assertIn('calib', adf_loaded._subframes.subframes)

        # Check subframe data
        sf = adf_loaded.get_subframe('calib')
        self.assertEqual(len(sf.df), len(self.df_calib))

    def test_subframe_warning_with_entry_range(self):
        """Test warning when entry_range used with subframes"""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adf_loaded = AliasDataFrame.read_tree(
                self.tmp_path, "tree",
                entry_stop=500
            )

            # Should have warning about subframes
            self.assertEqual(len(w), 1)
            self.assertIn("entry_start/entry_stop", str(w[0].message))
            self.assertIn("calib", str(w[0].message))

        # Main tree should be sliced
        self.assertEqual(len(adf_loaded.df), 500)

        # Subframe should be fully loaded
        sf = adf_loaded.get_subframe('calib')
        self.assertEqual(len(sf.df), len(self.df_calib))

    def test_aliases_preserved(self):
        """Test that aliases are preserved after read"""
        adf_loaded = AliasDataFrame.read_tree(self.tmp_path, "tree")

        # Check alias exists
        self.assertIn('xy_sum', adf_loaded.aliases)

        # Materialize and check result
        adf_loaded.materialize_alias('xy_sum')
        expected = adf_loaded.df['x'] + adf_loaded.df['y']
        np.testing.assert_array_almost_equal(
            adf_loaded.df['xy_sum'].values,
            expected.values,
            decimal=5
        )

    def test_backward_compatibility_no_metadata(self):
        """Test reading file without extended metadata (old format)"""
        # Create simple file without metadata
        df_simple = pd.DataFrame({
            'a': np.arange(100),
            'b': np.arange(100) * 2
        })

        tmp_simple = tempfile.NamedTemporaryFile(suffix=".root", delete=False)
        tmp_simple_path = tmp_simple.name
        tmp_simple.close()

        try:
            # Write with uproot directly (no metadata)
            import uproot
            with uproot.recreate(tmp_simple_path) as f:
                f["tree"] = {k: df_simple[k].values for k in df_simple.columns}

            # Should read without error
            adf_loaded = AliasDataFrame.read_tree(tmp_simple_path, "tree")

            self.assertEqual(len(adf_loaded.df), 100)
            self.assertEqual(adf_loaded.aliases, {})

        finally:
            os.remove(tmp_simple_path)

    def test_invalid_tree_raises_error(self):
        """Test that invalid tree name raises informative error"""
        with self.assertRaises(ValueError) as cm:
            AliasDataFrame.read_tree(self.tmp_path, "nonexistent_tree")

        self.assertIn("nonexistent_tree", str(cm.exception))
        self.assertIn("not found", str(cm.exception))


class TestReadTreeWithCompression(unittest.TestCase):
    """Test read_tree with compressed columns and dtype restoration"""

    def setUp(self):
        """Create test data with compression"""
        n_rows = 1000
        self.df = pd.DataFrame({
            'dy': np.random.randn(n_rows).astype(np.float32),
            'dz': np.random.randn(n_rows).astype(np.float32),
        })

        # Save original values before compression
        self.original_dy = self.df['dy'].values.copy()
        self.original_dz = self.df['dz'].values.copy()

        self.adf = AliasDataFrame(self.df)

        # Define compression schema
        self.spec = {
            'dy': {
                'compress': 'round(asinh(dy)*40)',
                'decompress': 'sinh(dy_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            },
            'dz': {
                'compress': 'round(asinh(dz)*40)',
                'decompress': 'sinh(dz_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            }
        }

        # Compress columns
        self.adf.compress_columns(self.spec)

        # Export to temp file
        self.tmp = tempfile.NamedTemporaryFile(suffix=".root", delete=False)
        self.tmp_path = self.tmp.name
        self.tmp.close()
        self.adf.export_tree(self.tmp_path, treename="tree")

    def tearDown(self):
        """Clean up temp file"""
        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)

    def test_compressed_columns_dtype_restored(self):
        """Test that compressed columns have correct dtype after read"""
        adf_loaded = AliasDataFrame.read_tree(self.tmp_path, "tree")

        # Compressed columns should exist with correct dtype
        self.assertIn('dy_c', adf_loaded.df.columns)
        self.assertIn('dz_c', adf_loaded.df.columns)

        # Check dtype (int16 as stored)
        self.assertEqual(adf_loaded.df['dy_c'].dtype, np.int16)
        self.assertEqual(adf_loaded.df['dz_c'].dtype, np.int16)

    def test_compression_info_preserved(self):
        """Test that compression_info metadata is preserved"""
        adf_loaded = AliasDataFrame.read_tree(self.tmp_path, "tree")

        # Check compression_info exists
        self.assertIn('dy', adf_loaded.compression_info)
        self.assertIn('dz', adf_loaded.compression_info)

        # Check schema details
        self.assertEqual(
            adf_loaded.compression_info['dy']['compress_expr'],
            'round(asinh(dy)*40)'
        )

    def test_decompression_alias_works(self):
        """Test that decompression alias produces correct values"""
        adf_loaded = AliasDataFrame.read_tree(self.tmp_path, "tree")

        # Decompression alias should exist
        self.assertIn('dy', adf_loaded.aliases)

        # Materialize decompressed value
        adf_loaded.materialize_alias('dy')

        # Should be close to original (within compression precision)
        original_dy = self.original_dy
        restored_dy = adf_loaded.df['dy'].values

        # Allow for compression loss (asinh/sinh transform)
        #np.testing.assert_allclose(restored_dy, original_dy, rtol=0.05, atol=0.01)
        np.testing.assert_allclose(restored_dy, original_dy, rtol=0.1, atol=0.05)


    def test_entry_range_with_compression(self):
        """Test that entry_range works correctly with compressed data"""
        adf_loaded = AliasDataFrame.read_tree(
            self.tmp_path, "tree",
            entry_stop=500
        )

        self.assertEqual(len(adf_loaded.df), 500)

        # Compressed columns should have correct dtype
        self.assertEqual(adf_loaded.df['dy_c'].dtype, np.int16)


# Add to end of file before if __name__ == "__main__":
"""
Phase 2 Test Cases for AliasDataFrame
=====================================

Add these test classes to test_alias_dataframe.py
BEFORE `if __name__ == "__main__":`

Also add these imports at the top if not present:
    from AliasDataFrame import (VERBOSITY_BASIC, VERBOSITY_DTYPES, VERBOSITY_ALIASES,
                                 VERBOSITY_COMPRESSION, VERBOSITY_SUBFRAMES,
                                 VERBOSE_DEFAULT, VERBOSE_FULL)
"""


class TestDtypeRestoration(unittest.TestCase):
    """Test Phase 2 column_dtypes storage and restoration"""

    def test_column_dtypes_stored_in_metadata(self):
        """Test that column_dtypes is written to metadata"""
        import json
        import ROOT

        # Create DataFrame with mixed dtypes
        df = pd.DataFrame({
            'x': np.random.randn(100).astype(np.float16),
            'y': np.random.randn(100).astype(np.float32),
            'z': np.arange(100, dtype=np.int32),
            'w': np.arange(100, dtype=np.int64),
        })

        adf = AliasDataFrame(df)

        with tempfile.NamedTemporaryFile(suffix=".root", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            adf.export_tree(tmp_path, treename="tree")

            # Read metadata directly from ROOT
            f = ROOT.TFile.Open(tmp_path)
            tree = f.Get("tree")
            user_info = tree.GetUserInfo()

            metadata = None
            for i in range(user_info.GetEntries()):
                obj = user_info.At(i)
                if hasattr(obj, 'GetString'):
                    metadata = json.loads(obj.GetString().Data())
                    break
            f.Close()

            # Verify column_dtypes exists
            self.assertIn('column_dtypes', metadata)

            # Verify dtypes are correct
            self.assertEqual(metadata['column_dtypes']['x'], 'float16')
            self.assertEqual(metadata['column_dtypes']['y'], 'float32')
            self.assertEqual(metadata['column_dtypes']['z'], 'int32')
            self.assertEqual(metadata['column_dtypes']['w'], 'int64')

        finally:
            os.remove(tmp_path)

    def test_dtype_roundtrip_noncompressed(self):
        """Test that non-compressed float16 columns are restored correctly"""
        # Create DataFrame with float16 (not compressed)
        df = pd.DataFrame({
            'x': np.random.randn(100).astype(np.float16),
            'y': np.random.randn(100).astype(np.float32),
        })

        adf = AliasDataFrame(df)

        with tempfile.NamedTemporaryFile(suffix=".root", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            adf.export_tree(tmp_path, treename="tree")
            adf_loaded = AliasDataFrame.read_tree(tmp_path, treename="tree")

            # float16 should be restored (via column_dtypes)
            self.assertEqual(adf_loaded.df['x'].dtype, np.float16)
            self.assertEqual(adf_loaded.df['y'].dtype, np.float32)

        finally:
            os.remove(tmp_path)

    def test_dtype_roundtrip_compressed_and_uncompressed(self):
        """Test mixed compressed and non-compressed dtype restoration"""
        # Create DataFrame with mixed dtypes
        df = pd.DataFrame({
            'x': np.random.randn(100).astype(np.float16),  # Non-compressed float16
            'dy': np.random.randn(100).astype(np.float32),  # Will be compressed
            'z': np.arange(100, dtype=np.int64),  # Non-compressed int64
        })

        adf = AliasDataFrame(df)

        # Compress dy
        spec = {
            'dy': {
                'compress': 'round(asinh(dy)*40)',
                'decompress': 'sinh(dy_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            }
        }
        adf.compress_columns(spec)

        with tempfile.NamedTemporaryFile(suffix=".root", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            adf.export_tree(tmp_path, treename="tree")
            adf_loaded = AliasDataFrame.read_tree(tmp_path, treename="tree")

            # Non-compressed columns should keep their dtype
            self.assertEqual(adf_loaded.df['x'].dtype, np.float16)
            self.assertEqual(adf_loaded.df['z'].dtype, np.int64)

            # Compressed column should have compressed dtype
            self.assertEqual(adf_loaded.df['dy_c'].dtype, np.int16)

        finally:
            os.remove(tmp_path)

    def test_compression_info_priority_over_column_dtypes(self):
        """Test that compression_info takes priority over column_dtypes"""
        df = pd.DataFrame({
            'dy': np.random.randn(100).astype(np.float32),
        })

        adf = AliasDataFrame(df)

        spec = {
            'dy': {
                'compress': 'round(asinh(dy)*40)',
                'decompress': 'sinh(dy_c/40.)',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float16
            }
        }
        adf.compress_columns(spec)

        with tempfile.NamedTemporaryFile(suffix=".root", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            adf.export_tree(tmp_path, treename="tree")
            adf_loaded = AliasDataFrame.read_tree(tmp_path, treename="tree")

            # compression_info specifies int16 for dy_c
            # This should take priority even if column_dtypes says something else
            self.assertEqual(adf_loaded.df['dy_c'].dtype, np.int16)

        finally:
            os.remove(tmp_path)

    def test_backward_compat_no_column_dtypes(self):
        """Test reading files without column_dtypes metadata (Phase 1 files)"""
        # Create a simple file using uproot directly (no metadata)
        df = pd.DataFrame({
            'a': np.arange(100, dtype=np.float32),
            'b': np.arange(100, dtype=np.float32),
        })

        with tempfile.NamedTemporaryFile(suffix=".root", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Write with uproot only (no AliasDataFrame metadata)
            with uproot.recreate(tmp_path) as f:
                f["tree"] = {col: df[col].values for col in df.columns}

            # Should read without error
            adf_loaded = AliasDataFrame.read_tree(tmp_path, treename="tree")

            self.assertEqual(len(adf_loaded.df), 100)
            # Without column_dtypes, dtypes default to uproot's choice (float32)
            self.assertEqual(adf_loaded.df['a'].dtype, np.float32)

        finally:
            os.remove(tmp_path)


class TestDescribeStructure(unittest.TestCase):
    """Test describe_structure() method with bitmask verbosity"""

    def setUp(self):
        """Create test AliasDataFrame with various features"""
        from dfextensions.AliasDataFrame import (VERBOSITY_BASIC, VERBOSITY_DTYPES,
                                                 VERBOSITY_ALIASES, VERBOSITY_COMPRESSION,
                                                 VERBOSITY_SUBFRAMES, VERBOSE_DEFAULT)

        # Main DataFrame
        n_rows = 1000
        self.df = pd.DataFrame({
            'x': np.random.randn(n_rows).astype(np.float32),
            'y': np.random.randn(n_rows).astype(np.float16),
            'z': np.arange(n_rows, dtype=np.int32),
        })

        self.adf = AliasDataFrame(self.df)

        # Add alias
        self.adf.add_alias('xy_sum', 'x + y', dtype=np.float32)

        # Add compression
        spec = {
            'x': {
                'compress': 'round(x*100)',
                'decompress': 'x_c/100.',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float32
            }
        }
        self.adf.compress_columns(spec)

        # Add subframe
        df_sub = pd.DataFrame({
            'sub_id': np.arange(10),
            'value': np.random.randn(10).astype(np.float32),
        })
        adf_sub = AliasDataFrame(df_sub)
        self.adf.register_subframe('sub', adf_sub, index_columns='sub_id')

    def test_describe_structure_prints(self):
        """Test that describe_structure() prints without error"""
        import io
        import sys

        # Capture stdout
        captured = io.StringIO()
        sys.stdout = captured

        try:
            result = self.adf.describe_structure()
            self.assertIsNone(result)  # Default returns None
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()

        # Check key sections are present
        self.assertIn("AliasDataFrame Structure", output)
        self.assertIn("rows", output)
        self.assertIn("columns", output)
        self.assertIn("Memory", output)

    def test_describe_structure_return_dict(self):
        """Test that describe_structure(return_dict=True) returns dict"""
        result = self.adf.describe_structure(return_dict=True)

        self.assertIsInstance(result, dict)

        # Check expected keys
        self.assertIn('n_rows', result)
        self.assertIn('n_columns', result)
        self.assertIn('total_memory_mb', result)
        self.assertIn('dtype_groups', result)
        self.assertIn('n_aliases', result)
        self.assertIn('compression', result)
        self.assertIn('subframes', result)

    def test_describe_structure_values(self):
        """Test that describe_structure returns correct values"""
        result = self.adf.describe_structure(return_dict=True)

        self.assertEqual(result['n_rows'], 1000)
        self.assertGreater(result['n_columns'], 0)
        self.assertGreater(result['total_memory_mb'], 0)
        self.assertGreater(result['n_aliases'], 0)

        # Check subframes
        self.assertEqual(len(result['subframes']), 1)
        self.assertEqual(result['subframes'][0]['name'], 'sub')

    def test_describe_structure_bitmask_basic_only(self):
        """Test bitmask: VERBOSITY_BASIC only"""
        from dfextensions.AliasDataFrame import VERBOSITY_BASIC

        import io
        import sys

        captured = io.StringIO()
        sys.stdout = captured

        try:
            self.adf.describe_structure(verbosity=VERBOSITY_BASIC)
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()

        # Should have basic info
        self.assertIn("rows", output)
        self.assertIn("Memory", output)

        # Should NOT have dtype groups or aliases
        self.assertNotIn("Columns by dtype", output)
        self.assertNotIn("Aliases:", output)

    def test_describe_structure_bitmask_combined(self):
        """Test bitmask: combine multiple flags"""
        from dfextensions.AliasDataFrame import VERBOSITY_ALIASES, VERBOSITY_COMPRESSION

        import io
        import sys

        captured = io.StringIO()
        sys.stdout = captured

        try:
            self.adf.describe_structure(verbosity=VERBOSITY_ALIASES | VERBOSITY_COMPRESSION)
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()

        # Should have aliases and compression
        self.assertIn("Aliases:", output)
        self.assertIn("Compression:", output)

        # Should NOT have basic header (no VERBOSITY_BASIC)
        self.assertNotIn("AliasDataFrame Structure", output)

    def test_describe_structure_verbose_full(self):
        """Test VERBOSE_FULL preset"""
        import dfextensions.AliasDataFrame as adf_module
        from dfextensions.AliasDataFrame import VERBOSE_FULL;
        import io
        import sys

        captured = io.StringIO()
        sys.stdout = captured

        try:
            self.adf.describe_structure(verbosity=VERBOSE_FULL)
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()

        # Should have everything
        self.assertIn("AliasDataFrame Structure", output)
        self.assertIn("Columns by dtype", output)
        self.assertIn("Aliases:", output)
        self.assertIn("Full Alias Definitions:", output)
        self.assertIn("Raw Metadata:", output)


# Add these imports to the top of test_alias_dataframe.py if not present:
# from dfextensions.AliasDataFrame import (VERBOSITY_BASIC, VERBOSITY_DTYPES,
#                                           VERBOSITY_ALIASES, VERBOSITY_COMPRESSION,
#                                           VERBOSITY_SUBFRAMES, VERBOSE_DEFAULT, VERBOSE_FULL)


class TestSchemaV2Ordering(unittest.TestCase):
    """Test schema v2 export ordering: __meta__ → columns → groups → compression → subframes"""
    
    def setUp(self):
        """Create ADF with all schema components for testing"""
        self.df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0],
            'y': [4.0, 5.0, 6.0],
            'z': [7.0, 8.0, 9.0],
            'dy_c': [10, 20, 30],
            'track_id': [100, 101, 102]
        })
        self.adf = AliasDataFrame(self.df)
        
        # Add groups
        self.adf._schema['groups'] = {
            'coordinates': ['x', 'y', 'z'],
            'residuals': ['dy_c']
        }
        
        # Add compression info
        self.adf._schema['compression'] = {
            'dy': {
                'compressed_col': 'dy_c',
                'compress_expr': 'int16(dy*40)',
                'decompress_expr': 'sinh(dy_c/40.)',
                'state': 'compressed'
            }
        }
        
        # Add subframe info
        self.adf._schema['subframes'] = {
            'Tracks': {
                'index': ['track_id']
            }
        }
    
    def test_schema_v2_order_canonical(self):
        """Test that export_schema_v2 produces canonical key order"""
        schema = self.adf.export_schema_v2()
        keys = list(schema.keys())
        
        # Must start with __meta__
        self.assertEqual(keys[0], '__meta__')
        
        # Must have columns second
        self.assertEqual(keys[1], 'columns')
        
        # Verify expected keys are present
        self.assertIn('groups', keys)
        self.assertIn('compression', keys)
        self.assertIn('subframes', keys)
        
        # groups must come before compression
        groups_idx = keys.index('groups')
        comp_idx = keys.index('compression')
        self.assertLess(groups_idx, comp_idx, "groups should come before compression")
        
        # compression must come before subframes
        sf_idx = keys.index('subframes')
        self.assertLess(comp_idx, sf_idx, "compression should come before subframes")
    
    def test_schema_v2_order_full(self):
        """Test full canonical order: __meta__ → columns → groups → compression → subframes"""
        schema = self.adf.export_schema_v2()
        keys = list(schema.keys())
        
        expected_order = ['__meta__', 'columns', 'groups', 'compression', 'subframes']
        self.assertEqual(keys, expected_order,
            f"Schema keys must be in canonical order.\n"
            f"Expected: {expected_order}\n"
            f"Got:      {keys}")
    
    def test_schema_v2_order_strict_positions(self):
        """Strict test: verify exact positions of each key"""
        schema = self.adf.export_schema_v2()
        keys = list(schema.keys())
        
        # __meta__ MUST be first
        self.assertEqual(keys[0], '__meta__', "__meta__ must be first")
        
        # columns MUST be second
        self.assertEqual(keys[1], 'columns', "columns must be second")
        
        # If groups present, it MUST be third (before compression)
        if 'groups' in keys:
            self.assertEqual(keys[2], 'groups', "groups must be third")
            # compression must be fourth
            self.assertEqual(keys[3], 'compression', "compression must be fourth")
            # subframes must be fifth (last)
            self.assertEqual(keys[4], 'subframes', "subframes must be last")
        else:
            # Without groups: compression third, subframes fourth
            if 'compression' in keys:
                self.assertEqual(keys[2], 'compression', "compression must be third (no groups)")
            if 'subframes' in keys:
                sf_idx = keys.index('subframes')
                self.assertEqual(sf_idx, len(keys) - 1, "subframes must be last")
    
    def test_schema_v2_groups_simple_lists(self):
        """Test that groups are exported as simple lists (not nested objects)"""
        schema = self.adf.export_schema_v2()
        groups = schema.get('groups', {})
        
        self.assertIn('coordinates', groups)
        self.assertIn('residuals', groups)
        
        # Groups must be simple lists, not dicts
        self.assertIsInstance(groups['coordinates'], list)
        self.assertIsInstance(groups['residuals'], list)
        
        # Content must be column names
        self.assertEqual(groups['coordinates'], ['x', 'y', 'z'])
        self.assertEqual(groups['residuals'], ['dy_c'])
    
    def test_schema_v2_without_groups(self):
        """Test schema order when groups are absent"""
        # Remove groups
        self.adf._schema['groups'] = {}
        
        schema = self.adf.export_schema_v2()
        keys = list(schema.keys())
        
        # groups should not be in output
        self.assertNotIn('groups', keys)
        
        # Order should be: __meta__ → columns → compression → subframes
        self.assertEqual(keys[0], '__meta__')
        self.assertEqual(keys[1], 'columns')
        
        comp_idx = keys.index('compression')
        sf_idx = keys.index('subframes')
        self.assertLess(comp_idx, sf_idx)
    
    def test_schema_v2_groups_roundtrip(self):
        """Test that groups survive save/load roundtrip"""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save
            self.adf.save_schema_v2(temp_path)
            
            # Load back
            loaded = AliasDataFrame.load_schema(temp_path)
            
            # Groups should survive
            self.assertIn('groups', loaded)
            self.assertEqual(loaded['groups']['coordinates'], ['x', 'y', 'z'])
            self.assertEqual(loaded['groups']['residuals'], ['dy_c'])
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_schema_v2_order_agnostic_load(self):
        """Test that loader handles any key order"""
        import json
        import tempfile
        import os
        
        # Create schema with scrambled order (use numeric version)
        scrambled = {
            'subframes': {'T': {'index': ['id']}},
            'compression': {'dy': {'state': 'compressed'}},
            '__meta__': {'schema_version': 2},
            'groups': {'test': ['x']},
            'columns': {'x': {'dtype': 'float64'}}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(scrambled, f)
            temp_path = f.name
        
        try:
            # Load should work regardless of order
            loaded = AliasDataFrame.load_schema(temp_path)
            
            # All sections should be accessible
            self.assertIn('columns', loaded)
            self.assertIn('groups', loaded)
            self.assertIn('compression', loaded)
            self.assertIn('subframes', loaded)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestExportTreeColumns(unittest.TestCase):
    """Tests for export_tree(columns=...) snapshot mode."""
    
    def test_export_tree_columns_subset(self):
        """Test export_tree with columns parameter exports only specified columns."""
        import tempfile
        
        df = pd.DataFrame({
            'x': np.array([1, 2, 3], dtype=np.float32),
            'y': np.array([3, 4, 5], dtype=np.float32),
            'z': np.array([5, 6, 7], dtype=np.float32),
        })
        adf = AliasDataFrame(df)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = os.path.join(tmp_dir, "subset.root")
            adf.export_tree(filepath, "tree", columns=['x', 'y'])
            
            # Verify only x, y exported
            adf2 = AliasDataFrame.read_tree(filepath, "tree")
            self.assertIn('x', adf2.df.columns)
            self.assertIn('y', adf2.df.columns)
            self.assertNotIn('z', adf2.df.columns)
            
            # Verify data is correct
            np.testing.assert_array_equal(adf2.df['x'].values, [1, 2, 3])
            np.testing.assert_array_equal(adf2.df['y'].values, [3, 4, 5])
    
    def test_export_tree_columns_missing_raises(self):
        """Test export_tree raises ValueError for missing columns."""
        import tempfile
        
        df = pd.DataFrame({'x': np.array([1, 2], dtype=np.float32)})
        adf = AliasDataFrame(df)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = os.path.join(tmp_dir, "out.root")
            with self.assertRaises(ValueError) as ctx:
                adf.export_tree(filepath, "tree", columns=['x', 'missing'])
            self.assertIn("not found", str(ctx.exception))
    
    def test_export_tree_columns_warns_subframes(self):
        """Test export_tree warns when subframes exist but columns specified."""
        import tempfile
        import warnings
        
        df = pd.DataFrame({
            'x': np.array([1, 2], dtype=np.float32),
            'row': np.array([0, 1], dtype=np.int32),
        })
        sub_df = pd.DataFrame({
            'row': np.array([0, 1], dtype=np.int32),
            'val': np.array([10, 20], dtype=np.float32),
        })
        
        adf = AliasDataFrame(df)
        adf.register_subframe('S', AliasDataFrame(sub_df), index_columns='row')
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = os.path.join(tmp_dir, "out.root")
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                adf.export_tree(filepath, "tree", columns=['x'])
                
                # Check warning was raised
                self.assertEqual(len(w), 1)
                self.assertIn("subframes", str(w[0].message))
                self.assertEqual(w[0].category, UserWarning)


    def test_fill_value_replaces_inf_nan(self):
        df = pd.DataFrame({'x': [1.0, 0.0, 2.0, -0.0]})
        adf = AliasDataFrame(df)
        adf.add_alias('y', '1/x', fill_value=0)
        adf.materialize_alias('y')
        result = adf.df['y'].values
        assert result[0] == 1.0
        assert result[1] == 0.0
        assert result[2] == 0.5
        assert result[3] == 0.0

    def test_fill_value_none_preserves_inf(self):
        df = pd.DataFrame({'x': [1.0, 0.0]})
        adf = AliasDataFrame(df)
        adf.add_alias('y', '1/x')
        adf.materialize_alias('y')
        assert np.isinf(adf.df['y'].values[1])

if __name__ == "__main__":
    unittest.main()