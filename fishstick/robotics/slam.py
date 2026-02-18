# =============================================================================
# Visual SLAM Algorithms
# =============================================================================


class VisualSLAM(ABC):
    """
    Abstract base class for Visual SLAM algorithms.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = SLAMState(current_pose=SE3Pose.identity())
        self.feature_extractor = FeatureExtractor(
            method=config.get("feature_method", "orb"),
            n_features=config.get("n_features", 1000),
        )
        self.frame_id = 0
        self.keyframe_id = 0

    @abstractmethod
    def track(self, image: Tensor, timestamp: float) -> SE3Pose:
        """
        Track camera pose from new image.

        Args:
            image: Input image [H, W] or [H, W, 3]
            timestamp: Frame timestamp

        Returns:
            Estimated camera pose
        """
        pass

    @abstractmethod
    def local_mapping(self):
        """Perform local bundle adjustment and map point culling."""
        pass


class ORBSLAM(VisualSLAM):
    """
    ORB-SLAM implementation (feature-based monocular/stereo SLAM).

    Based on: Mur-Artal et al. "ORB-SLAM2: An Open-Source SLAM System
    for Monocular, Stereo, and RGB-D Cameras", 2017.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.camera_matrix = config.get("camera_matrix", torch.eye(3))
        self.dist_coeffs = config.get("dist_coeffs", torch.zeros(4))
        self.baseline = config.get("baseline", 0.0)  # For stereo

        # Tracking states
        self.n_init_features = config.get("n_init_features", 100)
        self.min_keyframe_features = config.get("min_keyframe_features", 50)

        # Map
        self.map_points: Dict[int, Landmark] = {}
        self.keyframes: Dict[int, KeyFrame] = {}

        # Tracking variables
        self.prev_keyframe: Optional[KeyFrame] = None
        self.prev_image: Optional[Tensor] = None
        self.prev_keypoints: Optional[Tensor] = None
        self.prev_descriptors: Optional[Tensor] = None

        # ATLAS (multi-map support)
        self.atlases: List[Dict] = []
        self.current_atlas_id = 0

    def track(self, image: Tensor, timestamp: float) -> SE3Pose:
        """
        Track camera pose from new frame.

        Args:
            image: Input image
            timestamp: Frame timestamp

        Returns:
            Camera pose in world frame
        """
        self.frame_id += 1

        # Extract features
        keypoints, descriptors = self.feature_extractor.detect_and_compute(image)

        if len(keypoints) < self.min_keyframe_features:
            self.state.is_tracking = False
            return self.state.current_pose

        # Match with previous frame
        if self.prev_keypoints is not None and len(self.prev_keypoints) > 0:
            matches = self._match_descriptors(descriptors, self.prev_descriptors)

            # Estimate motion
            if len(matches) >= 8:
                pose = self._estimate_motion(keypoints, self.prev_keypoints, matches)
                self.state.current_pose = pose
            else:
                # Lost tracking
                self.state.is_tracking = False
        else:
            # First frame or reinitialization
            self.state.current_pose = SE3Pose.identity()

        # Check if new keyframe needed
        if self._need_new_keyframe(keypoints, self.state.current_pose):
            self._create_keyframe(image, keypoints, descriptors, timestamp)

        # Update tracking variables
        self.prev_image = image
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

        self.state.trajectory.append(self.state.current_pose)
        self.state.timestamp = timestamp

        return self.state.current_pose

    def _match_descriptors(
        self, desc1: Tensor, desc2: Tensor, ratio_threshold: float = 0.7
    ) -> List[Tuple[int, int]]:
        """
        Match descriptors using Lowe's ratio test.

        Args:
            desc1: Query descriptors [N, D]
            desc2: Reference descriptors [M, D]
            ratio_threshold: Ratio test threshold

        Returns:
            List of (query_idx, train_idx) matches
        """
        if len(desc1) == 0 or len(desc2) == 0:
            return []

        # Compute distances
        dists = torch.cdist(desc1, desc2)

        # Get two nearest neighbors
        dists_sorted, indices = torch.sort(dists, dim=1)

        # Ratio test
        matches = []
        for i in range(len(dists_sorted)):
            if dists_sorted[i, 0] < ratio_threshold * dists_sorted[i, 1]:
                matches.append((i, indices[i, 0].item()))

        return matches

    def _estimate_motion(
        self, kpts_curr: Tensor, kpts_prev: Tensor, matches: List[Tuple[int, int]]
    ) -> SE3Pose:
        """
        Estimate camera motion from feature matches.

        Args:
            kpts_curr: Current keypoints
            kpts_prev: Previous keypoints
            matches: Matched feature indices

        Returns:
            Relative camera pose
        """
        if len(matches) < 5:
            return self.state.current_pose

        # Get matched points
        matched_curr = torch.stack([kpts_curr[i] for i, _ in matches])
        matched_prev = torch.stack([kpts_prev[j] for _, j in matches])

        # Essential matrix estimation (simplified)
        # In practice, use RANSAC for robust estimation
        E = self._compute_essential_matrix(matched_curr, matched_prev)

        # Decompose to R and t
        R, t = self._decompose_essential(E)

        # Compose with previous pose
        current_R = self.state.current_pose.R
        current_t = self.state.current_pose.t

        new_R = torch.matmul(R, current_R)
        new_t = torch.matmul(R, current_t.unsqueeze(-1)).squeeze(-1) + t

        return SE3Pose(R=new_R, t=new_t)

    def _compute_essential_matrix(self, pts1: Tensor, pts2: Tensor) -> Tensor:
        """
        Compute essential matrix from point correspondences (8-point algorithm).
        """
        # Normalize points
        N = len(pts1)

        # Build constraint matrix
        A = torch.zeros(N, 9, device=pts1.device)
        A[:, 0] = pts2[:, 0] * pts1[:, 0]
        A[:, 1] = pts2[:, 0] * pts1[:, 1]
        A[:, 2] = pts2[:, 0]
        A[:, 3] = pts2[:, 1] * pts1[:, 0]
        A[:, 4] = pts2[:, 1] * pts1[:, 1]
        A[:, 5] = pts2[:, 1]
        A[:, 6] = pts1[:, 0]
        A[:, 7] = pts1[:, 1]
        A[:, 8] = 1.0

        # SVD
        _, _, Vt = torch.svd(A)
        E = Vt[-1].reshape(3, 3)

        # Enforce rank 2 constraint
        U, S, Vt = torch.svd(E)
        S = torch.diag(torch.tensor([1.0, 1.0, 0.0], device=E.device))
        E = torch.matmul(U, torch.matmul(S, Vt))

        return E

    def _decompose_essential(self, E: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Decompose essential matrix to rotation and translation.
        """
        U, S, Vt = torch.svd(E)

        # Ensure proper rotation
        if torch.det(U) < 0:
            U = -U
        if torch.det(Vt) < 0:
            Vt = -Vt

        # Translation is the last column of U
        t = U[:, 2]
        t = t / (torch.norm(t) + 1e-8)

        # Two possible rotations
        W = torch.tensor(
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]], device=E.device, dtype=E.dtype
        )
        R1 = torch.matmul(U, torch.matmul(W, Vt))
        R2 = torch.matmul(U, torch.matmul(W.T, Vt))

        # Choose rotation (simplified - should check cheirality)
        R = R1

        return R.unsqueeze(0), t.unsqueeze(0)

    def _need_new_keyframe(self, keypoints: Tensor, pose: SE3Pose) -> bool:
        """Check if a new keyframe is needed."""
        if self.prev_keyframe is None:
            return True

        # Distance from previous keyframe
        prev_pose = self.prev_keyframe.pose
        rel_pose = prev_pose.inverse().compose(pose)
        translation = torch.norm(rel_pose.t)

        # Check conditions
        too_far = translation > 0.1  # 10cm threshold
        too_few_features = len(keypoints) < self.min_keyframe_features * 1.5

        return too_far or too_few_features

    def _create_keyframe(
        self, image: Tensor, keypoints: Tensor, descriptors: Tensor, timestamp: float
    ):
        """Create a new keyframe."""
        keyframe = KeyFrame(
            id=self.keyframe_id,
            pose=self.state.current_pose,
            image=image,
            timestamp=timestamp,
            features=keypoints,
            descriptors=descriptors,
        )

        self.keyframes[self.keyframe_id] = keyframe
        self.prev_keyframe = keyframe
        self.keyframe_id += 1

    def local_mapping(self):
        """Perform local bundle adjustment and map management."""
        if len(self.keyframes) < 2:
            return

        # Get recent keyframes for local BA
        local_kf_ids = list(self.keyframes.keys())[-10:]  # Last 10 keyframes

        # Collect local map points
        local_points = set()
        for kf_id in local_kf_ids:
            for lm_id in self.keyframes[kf_id].landmarks:
                local_points.add(lm_id)

        # Perform local bundle adjustment
        self._local_bundle_adjustment(local_kf_ids, list(local_points))

    def _local_bundle_adjustment(
        self, keyframe_ids: List[int], landmark_ids: List[int]
    ):
        """
        Optimize local keyframes and landmarks.
        """
        # Collect parameters
        poses = []
        point_positions = []

        for kf_id in keyframe_ids:
            pose = self.keyframes[kf_id].pose
            # Convert to 6D vector (log map for rotation, translation)
            rot_vec = self._rotation_matrix_to_axis_angle(pose.R.squeeze(0))
            pose_vec = torch.cat([rot_vec, pose.t.squeeze(0)])
            poses.append(pose_vec)

        for lm_id in landmark_ids:
            if lm_id in self.map_points:
                point_positions.append(self.map_points[lm_id].position)

        if len(poses) == 0 or len(point_positions) == 0:
            return

        # Optimize (simplified - would use g2o or gtsam in practice)
        # Here we just do a few iterations of gradient descent
        for _ in range(10):
            # Compute residuals and gradients
            total_residual = 0.0

            for i, kf_id in enumerate(keyframe_ids):
                kf = self.keyframes[kf_id]
                for lm_id, lm in kf.landmarks.items():
                    if lm_id in landmark_ids:
                        # Project point to image
                        # Simplified projection
                        pass

        # Update poses and points
        for i, kf_id in enumerate(keyframe_ids):
            # Update keyframe pose
            pass

    def _rotation_matrix_to_axis_angle(self, R: Tensor) -> Tensor:
        """Convert rotation matrix to axis-angle representation."""
        # Simplified using trace formula
        trace = R.trace()
        angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))

        if angle.abs() < 1e-6:
            return torch.zeros(3, device=R.device)

        axis = torch.stack(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
        ) / (2 * torch.sin(angle))

        return axis * angle


class LSDSLAM(VisualSLAM):
    """
    LSD-SLAM implementation (direct monocular SLAM).

    Based on: Engel et al. "LSD-SLAM: Large-Scale Direct Monocular SLAM", 2014.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.camera = config.get("camera_matrix")
        self.min_grad_mag = config.get("min_grad_mag", 7.0)
        self.depth_init = config.get("depth_init", 1.0)

        # Semi-dense depth map
        self.reference_frame: Optional[Tensor] = None
        self.reference_pose: Optional[SE3Pose] = None
        self.depth_map: Optional[Tensor] = None
        self.depth_variance: Optional[Tensor] = None

        # Keyframe management
        self.keyframe_spacing = config.get("keyframe_spacing", 5)
        self.frames_since_keyframe = 0

    def track(self, image: Tensor, timestamp: float) -> SE3Pose:
        """Track using direct image alignment."""
        self.frame_id += 1

        # Convert to grayscale and float
        if image.dim() == 3:
            gray = (
                0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
            )
        else:
            gray = image

        gray = gray.float()

        if self.reference_frame is None:
            # Initialize first keyframe
            self._initialize_keyframe(gray)
            return self.state.current_pose

        # Compute photometric residual and Jacobian
        pose = self._direct_image_alignment(gray)

        # Update depth map with new observations
        self._update_depth_map(gray, pose)

        # Check if new keyframe needed
        self.frames_since_keyframe += 1
        if self.frames_since_keyframe >= self.keyframe_spacing:
            self._create_keyframe_from_depth(gray, pose)
            self.frames_since_keyframe = 0

        self.state.current_pose = pose
        self.state.trajectory.append(pose)
        self.state.timestamp = timestamp

        return pose

    def _initialize_keyframe(self, gray: Tensor):
        """Initialize first keyframe with uniform depth."""
        self.reference_frame = gray
        self.reference_pose = SE3Pose.identity()

        H, W = gray.shape
        self.depth_map = torch.ones(H, W, device=gray.device) * self.depth_init
        self.depth_variance = torch.ones(H, W, device=gray.device)

        # Select high-gradient pixels (edges)
        grad_x = torch.abs(gray[1:, :] - gray[:-1, :])
        grad_y = torch.abs(gray[:, 1:] - gray[:, :-1])

        grad_magnitude = torch.zeros_like(gray)
        grad_magnitude[1:, :] = torch.maximum(grad_magnitude[1:, :], grad_x)
        grad_magnitude[:, 1:] = torch.maximum(grad_magnitude[:, 1:], grad_y)

        # Mask for semi-dense tracking
        self.high_grad_mask = grad_magnitude > self.min_grad_mag

    def _direct_image_alignment(self, current_gray: Tensor) -> SE3Pose:
        """
        Align current frame to reference using photometric error.
        Uses Gauss-Newton optimization.
        """
        # Initial guess
        pose = self.state.current_pose

        # Get high gradient pixels
        y_coords, x_coords = torch.where(self.high_grad_mask)

        if len(y_coords) < 100:
            # Not enough features, use constant velocity model
            return pose

        # Sample pixels
        n_samples = min(2000, len(y_coords))
        indices = torch.randperm(len(y_coords))[:n_samples]
        y_samples = y_coords[indices]
        x_samples = x_coords[indices]

        # Get reference intensity and depth
        ref_intensity = self.reference_frame[y_samples, x_samples]
        ref_depth = self.depth_map[y_samples, x_samples]

        # Optimize pose
        for iteration in range(20):
            # Warp points
            warped_coords, valid = self._warp_points(
                x_samples.float(), y_samples.float(), ref_depth, pose
            )

            if valid.sum() < 50:
                break

            # Sample current image
            curr_intensity = self._sample_image(current_gray, warped_coords)

            # Compute residual
            residual = ref_intensity[valid] - curr_intensity

            # Compute Jacobian
            J = self._compute_jacobian(
                x_samples[valid], y_samples[valid], ref_depth[valid], pose
            )

            # Gauss-Newton update
            H = torch.matmul(J.T, J)
            b = torch.matmul(J.T, residual)

            try:
                delta = torch.linalg.solve(H, b)
            except:
                break

            # Update pose
            delta_pose = self._delta_to_pose(delta)
            pose = pose.compose(delta_pose.inverse())

            # Check convergence
            if torch.norm(delta) < 1e-6:
                break

        return pose

    def _warp_points(
        self, x: Tensor, y: Tensor, depth: Tensor, pose: SE3Pose
    ) -> Tuple[Tensor, Tensor]:
        """
        Warp 3D points from reference to current frame.

        Returns:
            warped_coords: [N, 2] image coordinates
            valid_mask: [N] boolean mask
        """
        # Back-project to 3D
        fx, fy = self.camera[0, 0], self.camera[1, 1]
        cx, cy = self.camera[0, 2], self.camera[1, 2]

        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth

        points_3d = torch.stack([X, Y, Z], dim=1)

        # Transform to current frame
        R = pose.R.squeeze(0)
        t = pose.t.squeeze(0)

        points_transformed = torch.matmul(points_3d, R.T) + t

        # Project to image
        Z_new = points_transformed[:, 2]
        x_new = points_transformed[:, 0] * fx / Z_new + cx
        y_new = points_transformed[:, 1] * fy / Z_new + cy

        # Check validity
        H, W = self.reference_frame.shape
        valid = (Z_new > 0.1) & (x_new >= 0) & (x_new < W) & (y_new >= 0) & (y_new < H)

        warped_coords = torch.stack([x_new, y_new], dim=1)

        return warped_coords, valid

    def _sample_image(self, image: Tensor, coords: Tensor) -> Tensor:
        """Bilinear image sampling."""
        # Simple nearest neighbor for efficiency
        x = coords[:, 0].long()
        y = coords[:, 1].long()
        return image[y, x]

    def _compute_jacobian(
        self, x: Tensor, y: Tensor, depth: Tensor, pose: SE3Pose
    ) -> Tensor:
        """
        Compute Jacobian of photometric error w.r.t. pose.
        """
        n_points = len(x)
        J = torch.zeros(n_points, 6, device=x.device)

        # Get image gradient
        fx, fy = self.camera[0, 0], self.camera[1, 1]
        cx, cy = self.camera[0, 2], self.camera[1, 2]

        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth

        # Image gradients (precomputed)
        grad_x = (
            self.reference_frame[
                y.long(), (x + 1).long().clamp(max=self.reference_frame.shape[1] - 1)
            ]
            - self.reference_frame[y.long(), (x - 1).long().clamp(min=0)]
        )
        grad_y = (
            self.reference_frame[
                (y + 1).long().clamp(max=self.reference_frame.shape[0] - 1), x.long()
            ]
            - self.reference_frame[(y - 1).long().clamp(min=0), x.long()]
        )

        # Jacobian of point w.r.t. pose
        J[:, 0] = grad_x * fx / Z
        J[:, 1] = grad_y * fy / Z
        J[:, 2] = -(grad_x * fx * X + grad_y * fy * Y) / (Z**2)
        J[:, 3] = -grad_x * fx * X * Y / Z - grad_y * fy * (1 + Y**2 / Z)
        J[:, 4] = grad_x * fx * (1 + X**2 / Z) + grad_y * fy * X * Y / Z
        J[:, 5] = -grad_x * fx * Y + grad_y * fy * X

        return J

    def _delta_to_pose(self, delta: Tensor) -> SE3Pose:
        """Convert 6D delta vector to SE3 pose."""
        omega = delta[:3]
        u = delta[3:]

        # Exponential map for rotation
        angle = torch.norm(omega)
        if angle < 1e-6:
            R = torch.eye(3, device=omega.device)
        else:
            axis = omega / angle
            K = torch.tensor(
                [
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0],
                ],
                device=omega.device,
            )
            R = (
                torch.eye(3, device=omega.device)
                + torch.sin(angle) * K
                + (1 - torch.cos(angle)) * torch.matmul(K, K)
            )

        # Translation
        t = u.unsqueeze(0)
        R = R.unsqueeze(0)

        return SE3Pose(R=R, t=t)

    def _update_depth_map(self, current_gray: Tensor, pose: SE3Pose):
        """Update depth map using stereo matching between keyframe and current frame."""
        y_coords, x_coords = torch.where(self.high_grad_mask)

        if len(y_coords) == 0:
            return

        # Sample subset
        n_samples = min(1000, len(y_coords))
        indices = torch.randperm(len(y_coords))[:n_samples]
        y_samples = y_coords[indices]
        x_samples = x_coords[indices]

        # Warp and compute depth error
        for i in range(len(y_samples)):
            y, x = y_samples[i].item(), x_samples[i].item()
            d = self.depth_map[y, x]

            # Warp point
            warped_coords, valid = self._warp_points(
                torch.tensor([x], device=current_gray.device).float(),
                torch.tensor([y], device=current_gray.device).float(),
                torch.tensor([d], device=current_gray.device),
                pose,
            )

            if valid[0]:
                # Epipolar search for better depth
                # Simplified: just update with small variation
                new_depth = d * (1 + 0.01 * (torch.rand(1).item() - 0.5))

                # Kalman update
                prior_var = self.depth_variance[y, x]
                measurement_var = 0.1

                K = prior_var / (prior_var + measurement_var)
                self.depth_map[y, x] = d + K * (new_depth - d)
                self.depth_variance[y, x] = (1 - K) * prior_var

    def _create_keyframe_from_depth(self, gray: Tensor, pose: SE3Pose):
        """Create new keyframe from current depth estimate."""
        self.reference_frame = gray
        self.reference_pose = pose

        # Regularization: reset variance for uncertain depths
        self.depth_variance[self.depth_variance > 10.0] = 10.0

        # Create keyframe
        keyframe = KeyFrame(
            id=self.keyframe_id, pose=pose, image=gray, timestamp=self.state.timestamp
        )

        self.keyframes[self.keyframe_id] = keyframe
        self.keyframe_id += 1

    def local_mapping(self):
        """Refine keyframe depth maps."""
        # Depth map regularization and fusion
        pass
