"""Export an English JSON and metric diagram of a provisional camera estimate."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import struct

import numpy as np

from ..common import ROOT, digest, load_yaml
from ..calibration.forward_kinematics import ForwardKinematics


DEFAULT_MESH_ROOT = Path('/home/kaixin/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/meshes')


def calibration_document(result, frames, resolution, *, camera_name='D435i'):
    if not frames:
        raise ValueError('No archived camera intrinsics')
    signatures = {json.dumps([f['K'], f['distortion'], f['serial'], f['optical_frame']]) for f in frames}
    if len(signatures) != 1:
        raise ValueError('Camera identity or intrinsics varied; cannot export one calibration')
    K = np.asarray(frames[0]['K'], float)
    T = np.asarray(result['T_torso_camera_optical']['matrix'], float)
    if K.shape != (3, 3) or T.shape != (4, 4) or not np.isfinite(np.r_[K.ravel(), T.ravel()]).all():
        raise ValueError('Invalid camera matrices')
    p, direction = T[:3, 3], T[:3, 2]
    coefficient_names = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6', 's1', 's2', 's3', 's4', 'tau_x', 'tau_y']
    count = len(frames[0]['distortion'])
    if count not in (4, 5, 8, 12, 14):
        raise ValueError('Unsupported OpenCV distortion vector length')
    document = dict(schema_version=1, camera_name=camera_name,
        camera_name_source='Operator identification and configured camera stream',
        serial_number=frames[0]['serial'], stream='color', length_unit='metre',
        status='provisional_estimate_not_accepted_for_calibration',
        image_size=dict(width=int(resolution[0]), height=int(resolution[1])),
        intrinsics=dict(source='Recorded color-camera packet metadata; not refitted in this run',
            camera_matrix=K.tolist(), fx=float(K[0, 0]), fy=float(K[1, 1]),
            cx=float(K[0, 2]), cy=float(K[1, 2]), focal_length_unit='pixel',
            distortion_coefficient_convention='OpenCV pinhole: '+', '.join(coefficient_names[:count]),
            distortion_model={4: 'opencv_brown_conrady', 5: 'opencv_brown_conrady',
                8: 'opencv_rational', 12: 'opencv_rational_with_thin_prism',
                14: 'opencv_rational_with_thin_prism_and_tilt'}[count],
            distortion_coefficients=frames[0]['distortion'],
            original_vendor_distortion_model='Not retained in the frame archive',
            image_preprocessing='Exported at the recorded resolution; no scaling or cropping applied'),
        extrinsics=dict(parent_frame=result['parent_frame'], child_frame=result['child_frame'],
            transform_name='T_torso_camera_optical',
            equation='p_torso = R_torso_camera_optical * p_camera_optical + t_torso_camera_optical',
            matrix_4x4=T.tolist(), rotation_matrix=T[:3, :3].tolist(),
            translation_m=dict(x=float(p[0]), y=float(p[1]), z=float(p[2])),
            quaternion_xyzw=result['T_torso_camera_optical']['quaternion_xyzw'],
            euler_extrinsic_xyz_deg=np.rad2deg(result['T_torso_camera_optical']['euler_xyz_rad']).tolist(),
            inverse_matrix_4x4=result['T_camera_optical_torso']['matrix'],
            torso_axis_convention=dict(x='forward', y='robot left', z='up'),
            optical_axis_convention=dict(x='image right', y='image down', z='viewing direction'),
            torso_origin='URDF torso_link frame at the waist pitch joint; not the chest centre',
            position_refers_to='Color-camera optical centre, not the camera housing',
            viewing_direction_in_torso=direction.tolist(),
            viewing_downward_angle_deg=float(np.rad2deg(np.arctan2(-direction[2], np.linalg.norm(direction[:2]))))),
        validation=dict(accepted=False, reference_foot_fixation_verified=False,
            physical_camera_pose_independently_measured=False,
            conditional_motion_clock_fit_passed=result['conditional_motion_clock_fit_passed'],
            exposure_time_alignment_independently_verified=False,
            held_out_consistency=result['held_out_validation'],
            sample_count=result['sample_count'], sampling_stable_count=result['sampling_stable_count'],
            warning='Consistency limits were not met. This estimate must not be treated as a validated calibration or a certified position accuracy.',
            assumptions=result['assumptions']),
        provenance=dict(created_utc=datetime.now(timezone.utc).isoformat(),
            source_session=result['source']['session'], source_frame_index_sha256=result['source']['frame_index_sha256'],
            timing_trace_manifest_sha256=result['source']['timing_trace_manifest_sha256'],
            urdf_sha256=result['source']['urdf_sha256'],
            reference_frame_used_for_hand_eye='left_ankle_roll_link',
            nominal_urdf_camera_extrinsics_used_as_a_fit_prior=False))
    if not json.dumps(document, allow_nan=False).isascii():
        raise ValueError('Export must contain English/ASCII text only')
    return document


def _stl(path):
    data = Path(path).read_bytes()
    count = struct.unpack('<I', data[80:84])[0]
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    if len(data) != 84+count*dtype.itemsize:
        raise ValueError('Expected a binary metric STL mesh')
    return np.frombuffer(data, dtype=dtype, count=count, offset=84)['vertices'].astype(float)


def draw(document, output, *, mesh_root, urdf, indices, file_prefix='D435i'):
    os.environ.setdefault('MPLCONFIGDIR', str(output/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial import ConvexHull

    T = np.array(document['extrinsics']['matrix_4x4'])
    p, R = T[:3, 3]*100., T[:3, :3]
    zmax = max(54., float(p[2])+12.)
    head_transform = ForwardKinematics(urdf, 'torso_link', 'head_link', indices).compute(np.zeros(29))
    torso = _stl(mesh_root/'torso_link.STL')*100.
    head = _stl(mesh_root/'head_link.STL')
    head = (head @ head_transform[:3, :3].T+head_transform[:3, 3])*100.
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig = plt.figure(figsize=(15, 9), facecolor='#f7f9fc', layout='constrained')
    grid = fig.add_gridspec(2, 3, width_ratios=[1.25, 1., 1.], height_ratios=[2.8, 1.])
    ax = fig.add_subplot(grid[:, 0], projection='3d', facecolor='#f7f9fc')
    colors = ['#52657a', '#a2afbd']
    for faces, color in zip((torso, head), colors):
        poly = Poly3DCollection(faces[::4], facecolor=color, edgecolor='none', alpha=.16)
        ax.add_collection3d(poly)
    ax.scatter(*p, color='#e57712', s=80, depthshade=False)
    ax.text(*(p+[0., 2., 3.]), document['camera_name']+' optical centre', color='#a04d00', fontsize=11)
    ax.scatter(0, 0, 0, color='#18283b', s=24)
    ax.text(0, 0, -4, 'torso_link origin', fontsize=10)
    for vector, label, color in zip(np.eye(3)*13., ('+X forward', '+Y left', '+Z up'), ('#c74d4d', '#318c66', '#3278b2')):
        ax.quiver(0, 0, 0, *vector, color=color, linewidth=2, arrow_length_ratio=.15)
        ax.text(*vector, label, color=color, fontsize=9)
    direction = R[:, 2]*22.
    ax.quiver(*p, *direction, color='#e57712', linewidth=2.5, arrow_length_ratio=.12)
    ax.text(*(p+direction), 'View', color='#a04d00')
    ax.plot([0, p[0], p[0], p[0]], [0, 0, p[1], p[1]], [0, 0, 0, p[2]], '--', color='#65748a', linewidth=1)
    ax.set(xlim=(-12, 31), ylim=(-18, 20), zlim=(-5, zmax), xlabel='X (cm)', ylabel='Y (cm)', zlabel='Z (cm)')
    ax.set_box_aspect((43, 38, zmax+5))
    ax.view_init(elev=18, azim=-48)
    ax.set_title('Camera pose in the torso frame', fontsize=13, pad=20)

    def silhouette(axis, dims):
        for mesh, color in zip((torso, head), colors):
            vertices = mesh.reshape(-1, 3)[:, dims]
            hull = ConvexHull(vertices)
            xy = vertices[hull.vertices]
            axis.fill(xy[:, 0], xy[:, 1], color=color, alpha=.22, linewidth=.7, edgecolor=color)
        axis.scatter(0, 0, s=22, color='#18283b', zorder=5)
        axis.grid(alpha=.18)
        axis.set_aspect('equal', adjustable='box')
        axis.set_ylim(-5, zmax)

    side = fig.add_subplot(grid[0, 1], facecolor='#f7f9fc')
    silhouette(side, [0, 2])
    side.scatter(p[0], p[2], s=70, color='#e57712', zorder=10)
    side.annotate('Optical centre', xy=(p[0], p[2]), xytext=(12, 48), color='#a04d00',
                  arrowprops=dict(arrowstyle='-', color='#a04d00'))
    side.annotate('', xy=(p[0]+direction[0], p[2]+direction[2]), xytext=(p[0], p[2]),
                  arrowprops=dict(arrowstyle='->', color='#e57712', lw=2))
    side.text(10, 22, f'View: {document["extrinsics"]["viewing_downward_angle_deg"]:.1f} deg down', color='#a04d00')
    side.plot([0, p[0], p[0]], [0, 0, p[2]], '--', color='#65748a', lw=1)
    side.annotate('', xy=(14, p[2]), xytext=(14, 0), arrowprops=dict(arrowstyle='<->', color='#3278b2'))
    side.text(15, 10, f'+Z = {p[2]:.1f} cm', rotation=90, color='#3278b2')
    side.text(2, -3.5, f'+X = {p[0]:.1f} cm', color='#c74d4d')
    side.set(xlim=(-12, 30), xlabel='+X: forward (cm)', ylabel='+Z: up (cm)', title='Side view (X-Z)')
    front = fig.add_subplot(grid[0, 2], facecolor='#f7f9fc')
    silhouette(front, [1, 2])
    front.scatter(p[1], p[2], s=70, color='#e57712', zorder=10)
    front.plot([0, p[1], p[1]], [0, 0, p[2]], '--', color='#65748a', lw=1)
    front.annotate('', xy=(p[1], 42), xytext=(0, 42), arrowprops=dict(arrowstyle='<->', color='#318c66'))
    front.text(-10, 45, f'+Y = {p[1]:.1f} cm (robot left)', color='#318c66')
    front.set(xlim=(-16, 16), xlabel='+Y: robot left (cm)', ylabel='+Z: up (cm)', title='Front view (Y-Z)')
    note = fig.add_subplot(grid[1, 1:])
    note.axis('off')
    quality = document['validation']['held_out_consistency']
    note.text(0., .98, 'PROVISIONAL ESTIMATE - VALIDATION LIMITS NOT MET', va='top',
              color='#a44621', weight='bold', fontsize=12)
    note.text(0., .77, f'Position relative to torso_link: [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}] cm\n'
        'The origin is at the waist pitch joint, not the chest centre.\n'
        'Grey geometry: URDF torso/head meshes. Orange: estimated optical centre and view.\n'
        f'Held-out consistency: {quality["translation_rms_mm"]:.1f} mm translation RMS; '
        f'{quality["rotation_rms_deg"]:.2f} deg rotation RMS.\n'
        'Foot fixation is unverified; motion-derived timing remains conditional.', va='top',
        linespacing=1.6, fontsize=10)
    fig.suptitle(document['camera_name']+' location relative to torso_link', fontsize=19, weight='bold')
    fig.savefig(output/f'{file_prefix}_torso_diagram.png', dpi=170, facecolor=fig.get_facecolor())
    fig.savefig(output/f'{file_prefix}_torso_diagram.svg', facecolor=fig.get_facecolor())
    plt.close(fig)


def export(result_directory, mesh_root, *, camera_name='D435i', file_prefix='D435i'):
    import cv2
    output = Path(result_directory).resolve()
    result = load_yaml(output/'result.yaml')
    session = Path(result['source']['session'])
    frames = [json.loads(line) for line in (session/'recording/frames.jsonl').read_text().splitlines()]
    if digest(session/'recording/frames.jsonl') != result['source']['frame_index_sha256']:
        raise ValueError('Source frames changed after estimation')
    image = cv2.imread(str(session/'recording'/frames[0]['image_file']))
    if image is None:
        raise ValueError('Recorded image is unavailable')
    if not file_prefix.replace('_', '').isalnum() or not file_prefix.isascii():
        raise ValueError('File prefix must contain ASCII letters, digits and underscores')
    if result.get('refitted_intrinsics'):
        refit = result['refitted_intrinsics']
        for frame in frames:
            frame['K'], frame['distortion'] = refit['camera_matrix'], refit['distortion_coefficients']
    document = calibration_document(result, frames, (image.shape[1], image.shape[0]), camera_name=camera_name)
    if result.get('refitted_intrinsics'):
        document['intrinsics']['source'] = 'Offline multi-view intrinsic refit from recorded board corners'
        if result['refitted_intrinsics'].get('feature_source', '').startswith('ArUco projective'):
            document['intrinsics']['source'] = 'Offline multi-view intrinsic refit using projective ArUco marker centres from raw images'
        document['intrinsics']['refit_validation'] = result['refitted_intrinsics']
        document['provenance']['chessboard_corners_redetected_with_fitted_intrinsics'] = bool(
            result.get('corners_redetected_with_refitted_intrinsics'))
    document['provenance']['estimate_yaml_sha256'] = digest(output/'result.yaml')
    robot = load_yaml(session/'recording/configuration.yaml')['robot']
    urdf = Path(robot['urdf'])
    if not urdf.is_absolute():
        urdf = ROOT/'config'/urdf
    draw(document, output, mesh_root=Path(mesh_root), urdf=urdf, indices=robot['joint_indices'], file_prefix=file_prefix)
    document['diagram_files'] = dict(png=f'{file_prefix}_torso_diagram.png', svg=f'{file_prefix}_torso_diagram.svg')
    destination = output/f'{file_prefix}_calibration.json'
    destination.write_text(json.dumps(document, indent=2, allow_nan=False, ensure_ascii=True)+'\n', encoding='ascii')
    print(f'English JSON: {destination}\nDiagram: {output/document["diagram_files"]["png"]}')
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-directory', type=Path, required=True)
    p.add_argument('--camera-name', default='D435i')
    p.add_argument('--file-prefix', default='D435i')
    p.add_argument('--mesh-root', type=Path, default=DEFAULT_MESH_ROOT)
    a = p.parse_args()
    export(a.result_directory, a.mesh_root, camera_name=a.camera_name, file_prefix=a.file_prefix)


if __name__ == '__main__':
    main()
