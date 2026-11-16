"""URDF tree FK equivalent to fixed-base Pinocchio, with no mesh dependency."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .transforms import transform, inverse
from ..common import digest


class ForwardKinematics:
    def __init__(self, urdf, base, mount, joint_indices):
        self.path = Path(urdf)
        self.sha256 = digest(urdf)
        root = ET.parse(urdf).getroot()
        self.base, self.mount = base, mount
        self.indices = joint_indices
        links = {x.get('name') for x in root.findall('link')}
        if base not in links or mount not in links:
            raise ValueError('Base/mount not present in actual URDF')
        self.joints = root.findall('joint')
        active = {j.get('name') for j in self.joints if j.get('type') != 'fixed'}
        if active != set(joint_indices) or len(active) != 29 or set(joint_indices.values()) != set(range(29)):
            raise ValueError('Require exact 29-DOF model and bijective SDK joint mapping')
        by_child = {j.find('child').get('link'): j for j in self.joints}
        def ancestors(link):
            result, seen = [link], set()
            while link in by_child:
                if link in seen:
                    raise ValueError('Cycle in URDF tree')
                seen.add(link)
                link = by_child[link].find('parent').get('link')
                result.append(link)
            return result

        base_ancestors, mount_ancestors = ancestors(base), ancestors(mount)
        common = next((link for link in base_ancestors if link in mount_ancestors), None)
        if common is None:
            raise ValueError('Base and mount must belong to the same URDF tree')

        def chain_to(link):
            chain = []
            while link != common:
                j = by_child[link]
                if j.get('type') not in ('fixed', 'revolute', 'continuous', 'prismatic') or j.find('mimic') is not None:
                    raise ValueError('Unsupported joint type/mimic in kinematic chain')
                chain.append(j)
                link = j.find('parent').get('link')
            return chain[::-1]

        self.base_chain, self.mount_chain = chain_to(base), chain_to(mount)
        self.common_ancestor = common
        # Keep the ancestor-base interface unchanged for existing pelvis plans.
        self.chain = self.mount_chain

    def compute(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (29,) or not np.isfinite(q).all():
            raise ValueError('Need 29 finite SDK joint angles in PR coordinates')
        return inverse(self._compute_chain(self.base_chain, q)) @ self._compute_chain(self.mount_chain, q)

    def _compute_chain(self, chain, q):
        T = np.eye(4)
        for j in chain:
            origin = j.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            T = T @ transform(Rotation.from_euler('xyz', rpy).as_matrix(), xyz)
            if j.get('type') == 'fixed':
                continue
            value = q[self.indices[j.get('name')]]
            limit = j.find('limit')
            if limit is not None and j.get('type') != 'continuous' and not float(limit.get('lower', '-inf'))-0.02 <= value <= float(limit.get('upper', 'inf'))+0.02:
                raise ValueError(f'Joint outside URDF limits: {j.get("name")}')
            a = j.find('axis')
            axis = np.fromstring(a.get('xyz') if a is not None else '1 0 0', sep=' ')
            axis /= np.linalg.norm(axis)
            T = T @ (transform(t=axis*value) if j.get('type') == 'prismatic' else transform(Rotation.from_rotvec(axis*value).as_matrix()))
        return T

    def compute_many(self, q):
        """Evaluate complete state windows without a Python loop over samples."""
        q = np.asarray(q, dtype=float)
        if q.ndim != 2 or q.shape[1] != 29 or not np.isfinite(q).all():
            raise ValueError('Need an N-by-29 finite PR joint array')
        def evaluate(chain):
            result = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
            for joint in chain:
                origin = joint.find('origin')
                xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
                rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
                result = result @ transform(Rotation.from_euler('xyz', rpy).as_matrix(), xyz)
                if joint.get('type') == 'fixed':
                    continue
                values = q[:, self.indices[joint.get('name')]]
                limit = joint.find('limit')
                if limit is not None and joint.get('type') != 'continuous' and (
                    np.any(values < float(limit.get('lower', '-inf'))-.02)
                    or np.any(values > float(limit.get('upper', 'inf'))+.02)):
                    raise ValueError(f'Joint outside URDF limits: {joint.get("name")}')
                axis_tag = joint.find('axis')
                axis = np.fromstring(axis_tag.get('xyz') if axis_tag is not None else '1 0 0', sep=' ')
                axis /= np.linalg.norm(axis)
                movement = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
                if joint.get('type') == 'prismatic':
                    movement[:, :3, 3] = values[:, None] * axis
                elif len(q):
                    movement[:, :3, :3] = Rotation.from_rotvec(values[:, None] * axis).as_matrix()
                result = result @ movement
            return result
        return np.linalg.inv(evaluate(self.base_chain)) @ evaluate(self.mount_chain)

    @property
    def chain_names(self):
        return ([f'inverse:{j.get("name")}' for j in self.base_chain[::-1]]
                + [j.get('name') for j in self.mount_chain])
