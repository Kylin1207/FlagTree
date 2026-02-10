# flagtree tle
import pytest

import triton.experimental.tle as tle
from triton.experimental.tle.distributed import _mesh_to_cluster_dims, _normalize_remote_shard_id
import triton.language.core as tlcore


class TestDeviceMesh:

    def test_device_mesh_shape_and_flatten(self):
        mesh = tle.device_mesh(
            {
                "node": [("node_x", 2), ("node_y", 2)],
                "device": 4,
                "block_cluster": [("cluster_x", 2), ("cluster_y", 2)],
                "block": 4,
            }
        )
        assert mesh.shape == (2, 2, 4, 2, 2, 4)
        assert mesh.ndim == 6
        assert mesh.size == 256

        flat = mesh.flatten()
        assert flat.shape == (256,)
        assert flat.dim_names == ("flat",)

    def test_device_mesh_slice_submesh(self):
        mesh = tle.device_mesh(
            {
                "node": [("node_x", 2), ("node_y", 2)],
                "device": 4,
            }
        )
        sub = mesh[1, :, 2]
        assert sub.shape == (2,)
        assert sub.dim_names == ("node_y",)
        assert sub.size == 2

    def test_device_mesh_invalid_topology(self):
        with pytest.raises(TypeError):
            tle.device_mesh({"node": "2"})
        with pytest.raises(ValueError):
            tle.device_mesh({"node": []})
        with pytest.raises(ValueError):
            tle.device_mesh({"node": [("x", 0)]})


class TestShardingSpec:

    def test_sharding_spec_states(self):
        mesh = tle.device_mesh(
            {
                "device": 4,
                "cluster": [("cluster_x", 2), ("cluster_y", 2)],
                "block": 4,
            }
        )
        spec = tle.sharding(mesh, split=[["cluster_x", "cluster_y"], "device"], partial=["block"])
        assert spec.axis_state("cluster_x") == "S"
        assert spec.axis_state("cluster_y") == "S"
        assert spec.axis_state("device") == "S"
        assert spec.axis_state("block") == "P"
        assert spec.broadcast == tuple()

    def test_sharding_rejects_overlap(self):
        mesh = tle.device_mesh({"device": 4, "block": 4})
        with pytest.raises(ValueError):
            tle.sharding(mesh, split=["device"], partial=["device"])

    def test_make_sharded_tensor(self):
        mesh = tle.device_mesh({"device": 4, "block": 4})
        spec = tle.sharding(mesh, split=["device", "block"], partial=[])
        st = tle.make_sharded_tensor("x_ptr", sharding=spec, shape=[4, 8])
        assert st.handle == "x_ptr"
        assert st.shape == (4, 8)
        assert st.sharding == spec

    def test_reshard_deferred(self):
        mesh = tle.device_mesh({"device": 4})
        spec = tle.sharding(mesh, split=["device"], partial=[])
        st = tle.make_sharded_tensor("x_ptr", sharding=spec, shape=[8])
        with pytest.raises(NotImplementedError):
            tle.reshard(st, spec)


class TestRemoteShardId:

    def test_normalize_remote_shard_id_scalar(self):
        assert _normalize_remote_shard_id(3, None) == 3
        with pytest.raises(ValueError):
            _normalize_remote_shard_id(-1, None)

    def test_normalize_remote_shard_id_tuple(self):
        mesh = tle.device_mesh({"cluster": [("x", 2), ("y", 4)]})
        assert _normalize_remote_shard_id((1, 3), mesh) == 7
        with pytest.raises(ValueError):
            _normalize_remote_shard_id((2, 0), mesh)
        with pytest.raises(ValueError):
            _normalize_remote_shard_id((1, 3), None)

    def test_m3_entrypoints_are_builtins(self):
        assert tlcore.is_builtin(tle.remote)
        assert tlcore.is_builtin(tle.distributed_barrier)


class TestClusterDims:

    def test_mesh_to_cluster_dims_prefers_cluster_axes(self):
        mesh = tle.device_mesh(
            {
                "node": [("node_x", 2), ("node_y", 2)],
                "device": 4,
                "block_cluster": [("cluster_x", 2), ("cluster_y", 1)],
                "block": 8,
            }
        )
        assert _mesh_to_cluster_dims(mesh) == (2, 1, 1)

    def test_mesh_to_cluster_dims_fallback_to_block_axes(self):
        mesh = tle.device_mesh({"device": 4, "block": [("block_x", 2), ("block_y", 2)]})
        assert _mesh_to_cluster_dims(mesh) == (2, 2, 1)
