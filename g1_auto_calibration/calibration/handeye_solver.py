"""G_i X C_i = Y; X=T_mount_camera, Y=T_base_board. No guessed extrinsics."""
import itertools
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .transforms import checked, transform, inverse, mean_pose, pose_errors


def motions(G, C):
    # G_i X C_i = G_j X C_j => inv(G_j) G_i X = X C_j inv(C_i).
    return [(inverse(G[j]) @ G[i], C[j] @ inverse(C[i]))
            for i, j in itertools.combinations(range(len(G)), 2)]


def observability(G, C):
    pairs = motions(G, C)
    vectors = np.array([Rotation.from_matrix(a[:3, :3]).as_rotvec() for a, _ in pairs])
    keep = np.linalg.norm(vectors, axis=1) > np.deg2rad(2)
    if keep.sum() < 3:
        raise ValueError('Insufficient rotational excitation: need diverse settled poses')
    axes = vectors[keep]
    s = np.linalg.svd(axes, compute_uv=False)
    trans = np.concatenate([a[:3, :3]-np.eye(3) for a, _ in pairs])
    st = np.linalg.svd(trans, compute_uv=False)
    if s[1] < .05 or s[1]/s[0] < .05 or st[-1]/st[0] < .03:
        raise ValueError('Degenerate/ill-conditioned single-axis motion: full 6-DoF is unobservable')
    return dict(rotation_singular_values=s.tolist(), translation_singular_values=st.tolist(),
                rotational_axis_ratio=float(s[1]/s[0]), translation_condition_number=float(st[0]/st[-1]))


def linear_fit(G, C, method):
    pairs = motions(G, C)
    if method == 'svd_log':
        aa, bb = [], []
        for A, B in pairs:
            a = Rotation.from_matrix(A[:3, :3]).as_rotvec()
            b = Rotation.from_matrix(B[:3, :3]).as_rotvec()
            if min(np.linalg.norm(a), np.linalg.norm(b)) > .01:
                aa.append(a)
                bb.append(b)
        R = Rotation.align_vectors(np.array(aa), np.array(bb))[0].as_matrix()
    elif method == 'kronecker':
        M = np.concatenate([np.kron(np.eye(3), A[:3, :3])-np.kron(B[:3, :3].T, np.eye(3)) for A, B in pairs])
        raw = np.linalg.svd(M, full_matrices=False)[2][-1].reshape(3, 3, order='F')
        if np.linalg.det(raw) < 0:
            raw = -raw
        u, _, v = np.linalg.svd(raw)
        R = u @ np.diag([1, 1, np.linalg.det(u @ v)]) @ v
    elif method.startswith('opencv_'):
        if not hasattr(cv2, 'calibrateHandEye'):
            raise RuntimeError('Installed OpenCV lacks calibrateHandEye; equivalent solvers are available')
        flag = getattr(cv2, 'CALIB_HAND_EYE_'+method[7:].upper())
        R, t = cv2.calibrateHandEye([g[:3, :3] for g in G], [g[:3, 3] for g in G],
                                  [c[:3, :3] for c in C], [c[:3, 3] for c in C], method=flag)
        return checked(transform(R, t))
    else:
        raise ValueError(method)
    L = np.concatenate([A[:3, :3]-np.eye(3) for A, _ in pairs])
    rhs = np.concatenate([R @ B[:3, 3]-A[:3, 3] for A, B in pairs])
    t = np.linalg.lstsq(L, rhs, rcond=None)[0]
    return checked(transform(R, t))


def pack(T):
    return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def unpack(x):
    return transform(Rotation.from_rotvec(x[:3]).as_matrix(), x[3:6])


def refine(G, C, X):
    Y = mean_pose([g @ X @ c for g, c in zip(G, C)])

    def residual(p):
        x, y = unpack(p[:6]), unpack(p[6:])
        ds = [inverse(y) @ g @ x @ c for g, c in zip(G, C)]
        return np.concatenate([np.r_[d[:3, 3]/.005, Rotation.from_matrix(d[:3, :3]).as_rotvec()/np.deg2rad(.5)] for d in ds])

    fit = least_squares(residual, np.r_[pack(X), pack(Y)], loss='soft_l1', max_nfev=300)
    if not fit.success:
        raise ValueError('Nonlinear refinement failed to converge')
    return checked(unpack(fit.x[:6]))


def solve(G, C, *, refine_result=True, seed=42):
    G, C = [checked(g) for g in G], [checked(c) for c in C]
    if len(G) != len(C) or len(G) < 15:
        raise ValueError('Need at least 15 valid independent poses; recommend 20-30')
    obs = observability(G, C)
    # Holdout is fixed before fitting, outlier selection or method comparison.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(G))
    hold = sorted(order[:max(4, len(G)//5)].tolist())
    train = sorted(set(range(len(G)))-set(hold))
    methods = ['svd_log', 'kronecker']
    if hasattr(cv2, 'calibrateHandEye'):
        methods += ['opencv_'+m for m in ['tsai', 'park', 'horaud', 'andreff', 'daniilidis']]
    reports, candidates = {}, []
    for method in methods:
        try:
            tg, tc = [G[i] for i in train], [C[i] for i in train]
            observability(tg, tc)
            # Consensus fitted ONLY on training poses, then refit on its inliers.
            best = None
            for iteration in range(61):
                subset = np.arange(len(train)) if iteration == 0 else rng.choice(len(train), 5, replace=False)
                try:
                    sg, sc = [tg[i] for i in subset], [tc[i] for i in subset]
                    observability(sg, sc)
                    x = linear_fit(sg, sc, method)
                    ys = [g @ x @ c for g, c in zip(tg, tc)]
                    y = mean_pose([ys[i] for i in subset])
                    mm, deg = pose_errors(y, ys)
                    mask = (mm < 15) & (deg < 2)
                    score = (int(mask.sum()), -float(np.median(mm+deg*5)))
                    if best is None or score > best[0]:
                        best = score, mask
                except (ValueError, RuntimeError, np.linalg.LinAlgError, cv2.error):
                    continue
            if best is None or best[1].sum() < max(8, int(np.ceil(.6*len(train)))):
                raise ValueError('No adequate training consensus')
            accepted = [train[i] for i in np.flatnonzero(best[1])]
            ag, ac = [G[i] for i in accepted], [C[i] for i in accepted]
            observability(ag, ac)
            x = linear_fit(ag, ac, method)
            if refine_result:
                x = refine(ag, ac, x)
            y = mean_pose([G[i] @ x @ C[i] for i in accepted])
            mm, deg = pose_errors(y, [G[i] @ x @ C[i] for i in accepted])
            # Selection by training residual, never by held-out residual.
            score = float(np.mean(mm/5+deg/.5))
            reports[method] = dict(status='candidate', train_indices=accepted,
                rejected_train_indices=sorted(set(train)-set(accepted)), train_translation_rms_mm=float(np.sqrt(np.mean(mm**2))),
                train_rotation_rms_deg=float(np.sqrt(np.mean(deg**2))), T_mount_camera=x.tolist())
            candidates.append((score, method, x, y, accepted))
        except (ValueError, RuntimeError, np.linalg.LinAlgError, cv2.error) as exc:
            reports[method] = dict(status='failed', reason=str(exc))
    if not candidates:
        raise ValueError(f'All hand-eye methods failed: {reports}')
    _, method, x, y, accepted = min(candidates, key=lambda v: v[0])
    return x, y, dict(method=method, algorithms=reports, observability=obs,
        training_indices=train, accepted_train_indices=accepted, held_out_indices=hold,
        rejected_train_indices=sorted(set(train)-set(accepted)), input_samples=len(G),
        refined=refine_result, seed=seed, convention='G=T_base_mount; C=T_camera_board; X=T_mount_camera; G X C = T_base_board')
