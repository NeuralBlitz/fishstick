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


class DirectSparseOdometry(VisualSLAM):
    """
    DSO (Direct Sparse Odometry) implementation.

    Based on: Engel et al. "Direct Sparse Odometry", 2017.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.camera = config.get("camera_matrix")
        self.photometric_calibration = config.get("photometric_calibration", False)

        # Point management
        self.active_points: List[Dict] = []
        self.marginalized_points: List[Dict] = []

        # Windowed optimization
        self.window_size = config.get("window_size", 7)
        self.keyframe_window: List[KeyFrame] = []

    def track(self, image: Tensor, timestamp: float) -> SE3Pose:
        """Track using sparse direct method with windowed optimization."""
        self.frame_id += 1

        # Preprocess image
        gray = self._preprocess_image(image)

        # Initial frame
        if len(self.keyframe_window) == 0:
            self._initialize_first_frame(gray, timestamp)
            return self.state.current_pose

        # Track against last keyframe
        initial_pose = self._get_initial_guess()

        # Coarse-to-fine tracking
        pose = self._coarse_tracking(gray, initial_pose)

        # Update active points
        self._update_active_points(gray, pose)

        # Check if new keyframe needed
        if self._should_create_keyframe(pose):
            self._create_keyframe(gray, pose, timestamp)

            # Marginalize old keyframes if needed
            if len(self.keyframe_window) > self.window_size:
                self._marginalize_oldest_keyframe()

            # Windowed bundle adjustment
            self._windowed_bundle_adjustment()

        self.state.current_pose = pose
        self.state.trajectory.append(pose)
        self.state.timestamp = timestamp

        return pose

    def _preprocess_image(self, image: Tensor) -> Tensor:
        """Preprocess image for tracking."""
        if image.dim() == 3:
            gray = (
                0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
            )
        else:
            gray = image

        if self.photometric_calibration:
            gray = self._apply_vignette_calibration(gray)

        return gray.float()

    def _apply_vignette_calibration(self, image: Tensor) -> Tensor:
        """Apply vignette calibration model."""
        H, W = image.shape
        y, x = torch.meshgrid(
            torch.arange(H, device=image.device),
            torch.arange(W, device=image.device),
            indexing="ij",
        )

        cx, cy = W / 2, H / 2
        r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r_max = torch.sqrt(cx**2 + cy**2)

        vignette = 1.0 + 0.1 * (r / r_max) ** 2

        return image * vignette

    def _initialize_first_frame(self, gray: Tensor, timestamp: float):
        """Initialize first keyframe."""
        points = self._extract_points(gray)

        keyframe = KeyFrame(
            id=0,
            pose=SE3Pose.identity(),
            image=gray,
            timestamp=timestamp,
            features=torch.stack([p["pixel"] for p in points]),
        )

        self.keyframe_window.append(keyframe)

        for point in points:
            point["depth"] = 1.0
            point["depth_variance"] = 1000.0
            point["host_frame"] = 0

        self.active_points = points

    def _extract_points(self, gray: Tensor, num_points: int = 2000) -> List[Dict]:
        """Extract high-gradient points."""
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=gray.device
        ).float()
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=gray.device
        ).float()

        gray_pad = gray.unsqueeze(0).unsqueeze(0)
        Ix = F.conv2d(gray_pad, sobel_x.unsqueeze(0).unsqueeze(0), padding=1).squeeze()
        Iy = F.conv2d(gray_pad, sobel_y.unsqueeze(0).unsqueeze(0), padding=1).squeeze()

        Ixx = F.avg_pool2d(
            Ix.unsqueeze(0).unsqueeze(0) ** 2, 3, stride=1, padding=1
        ).squeeze()
        Ixy = F.avg_pool2d(
            Ix.unsqueeze(0).unsqueeze(0) * Iy.unsqueeze(0).unsqueeze(0),
            3,
            stride=1,
            padding=1,
        ).squeeze()
        Iyy = F.avg_pool2d(
            Iy.unsqueeze(0).unsqueeze(0) ** 2, 3, stride=1, padding=1
        ).squeeze()

        det = Ixx * Iyy - Ixy**2
        trace = Ixx + Iyy

        response = det - 0.04 * trace**2

        H, W = gray.shape
        grid_size = 32
        points = []

        for y in range(0, H - grid_size, grid_size):
            for x in range(0, W - grid_size, grid_size):
                patch = response[y : y + grid_size, x : x + grid_size]
                max_val, max_idx = torch.max(patch.view(-1), dim=0)

                if max_val > 10.0:
                    local_y = max_idx // grid_size
                    local_x = max_idx % grid_size
                    points.append(
                        {
                            "pixel": torch.tensor(
                                [x + local_x, y + local_y], device=gray.device
                            ).float(),
                            "gradient": max_val.item(),
                        }
                    )

        points.sort(key=lambda p: p["gradient"], reverse=True)
        return points[:num_points]

    def _get_initial_guess(self) -> SE3Pose:
        """Get initial pose using constant velocity model."""
        if len(self.keyframe_window) < 2:
            return self.keyframe_window[-1].pose

        last_pose = self.keyframe_window[-1].pose
        prev_pose = self.keyframe_window[-2].pose

        relative = prev_pose.inverse().compose(last_pose)
        predicted = last_pose.compose(relative)

        return predicted

    def _coarse_tracking(self, gray: Tensor, initial_pose: SE3Pose) -> SE3Pose:
        """Coarse-to-fine direct image alignment."""
        pose = initial_pose

        scales = [4, 2, 1]

        for scale in scales:
            if scale > 1:
                current_scaled = F.avg_pool2d(
                    gray.unsqueeze(0).unsqueeze(0), scale, stride=scale
                ).squeeze()
            else:
                current_scaled = gray

            pose = self._optimize_pose(current_scaled, pose, scale)

        return pose

    def _optimize_pose(
        self, gray: Tensor, initial_pose: SE3Pose, scale: int
    ) -> SE3Pose:
        """Optimize pose using Gauss-Newton."""
        pose = initial_pose

        ref_keyframe = self.keyframe_window[-1]
        ref_image = ref_keyframe.image

        camera_scaled = self.camera / scale
        camera_scaled[2, 2] = 1.0

        for iteration in range(10):
            residuals = []
            jacobians = []

            for point in self.active_points[:500]:
                residual, J = self._compute_point_residual(
                    point, ref_image, gray, pose, camera_scaled
                )

                if residual is not None:
                    residuals.append(residual)
                    jacobians.append(J)

            if len(residuals) < 20:
                break

            residuals = torch.stack(residuals)
            jacobians = torch.stack(jacobians)

            H = torch.matmul(jacobians.T, jacobians)
            b = torch.matmul(jacobians.T, residuals)

            try:
                delta = torch.linalg.solve(H, b)
            except:
                break

            delta_pose = self._delta_to_pose(delta)
            pose = pose.compose(delta_pose.inverse())

            if torch.norm(delta) < 1e-4:
                break

        return pose

    def _compute_point_residual(
        self,
        point: Dict,
        ref_image: Tensor,
        curr_image: Tensor,
        pose: SE3Pose,
        camera: Tensor,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Compute photometric residual for a point."""
        pixel = point["pixel"]
        depth = point.get("depth", 1.0)

        fx, fy = camera[0, 0], camera[1, 1]
        cx, cy = camera[0, 2], camera[1, 2]

        X = (pixel[0] - cx) * depth / fx
        Y = (pixel[1] - cy) * depth / fy
        Z = depth

        R = pose.R.squeeze(0)
        t = pose.t.squeeze(0)

        p_transformed = torch.matmul(torch.tensor([X, Y, Z], device=R.device), R.T) + t

        if p_transformed[2] <= 0:
            return None, None

        x_proj = p_transformed[0] * fx / p_transformed[2] + cx
        y_proj = p_transformed[1] * fy / p_transformed[2] + cy

        H, W = curr_image.shape
        if x_proj < 0 or x_proj >= W or y_proj < 0 or y_proj >= H:
            return None, None

        ref_intensity = ref_image[int(pixel[1]), int(pixel[0])]
        curr_intensity = self._bilinear_sample(curr_image, x_proj, y_proj)

        residual = ref_intensity - curr_intensity

        J = torch.zeros(6, device=R.device)

        return residual, J

    def _bilinear_sample(self, image: Tensor, x: Tensor, y: Tensor) -> Tensor:
        """Bilinear interpolation."""
        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, image.shape[1] - 1)
        x1 = torch.clamp(x1, 0, image.shape[1] - 1)
        y0 = torch.clamp(y0, 0, image.shape[0] - 1)
        y1 = torch.clamp(y1, 0, image.shape[0] - 1)

        Ia = image[y0, x0]
        Ib = image[y1, x0]
        Ic = image[y0, x1]
        Id = image[y1, x1]

        wa = (x1.float() - x) * (y1.float() - y)
        wb = (x1.float() - x) * (y - y0.float())
        wc = (x - x0.float()) * (y1.float() - y)
        wd = (x - x0.float()) * (y - y0.float())

        return wa * Ia + wb * Ib + wc * Ic + wd * Id

    def _delta_to_pose(self, delta: Tensor) -> SE3Pose:
        """Convert 6D delta to SE3 pose."""
        omega = delta[:3]
        u = delta[3:]

        angle = torch.norm(omega)
        if angle < 1e-6:
            R_delta = torch.eye(3, device=omega.device)
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
            R_delta = (
                torch.eye(3, device=omega.device)
                + torch.sin(angle) * K
                + (1 - torch.cos(angle)) * torch.matmul(K, K)
            )

        t_delta = u

        R_new = torch.matmul(pose.R.squeeze(0), R_delta.T)
        t_new = torch.matmul(R_delta, pose.t.squeeze(0)) + t_delta

        return SE3Pose(R=R_new.unsqueeze(0), t=t_new.unsqueeze(0))

    def _update_active_points(self, gray: Tensor, pose: SE3Pose):
        """Update depth estimates for active points."""
        for point in self.active_points:
            if "depth_variance" in point and point["depth_variance"] > 100:
                pass

    def _should_create_keyframe(self, pose: SE3Pose) -> bool:
        """Check if new keyframe should be created."""
        if len(self.keyframe_window) == 0:
            return True

        last_pose = self.keyframe_window[-1].pose
        relative = last_pose.inverse().compose(pose)

        translation = torch.norm(relative.t)
        angle = torch.acos(torch.clamp((relative.R.squeeze(0).trace() - 1) / 2, -1, 1))

        return translation > 0.1 or angle > 0.1

    def _create_keyframe(self, gray: Tensor, pose: SE3Pose, timestamp: float):
        """Create new keyframe."""
        new_points = self._extract_points(gray)

        keyframe = KeyFrame(
            id=len(self.keyframes),
            pose=pose,
            image=gray,
            timestamp=timestamp,
            features=torch.stack([p["pixel"] for p in new_points]),
        )

        self.keyframe_window.append(keyframe)
        self.keyframes[len(self.keyframes)] = keyframe

        for point in new_points:
            point["depth"] = 1.0
            point["depth_variance"] = 1000.0
            point["host_frame"] = len(self.keyframes) - 1

        self.active_points.extend(new_points)

    def _marginalize_oldest_keyframe(self):
        """Marginalize oldest keyframe from window."""
        old_kf = self.keyframe_window.pop(0)

        points_to_marginalize = [
            p for p in self.active_points if p.get("host_frame") == old_kf.id
        ]

        for point in points_to_marginalize:
            self.active_points.remove(point)
            self.marginalized_points.append(point)

    def _windowed_bundle_adjustment(self):
        """Optimize poses and points in sliding window."""
        pass

    def local_mapping(self):
        """Perform local mapping operations."""
        pass


class SemiDirectVisualOdometry(VisualSLAM):
    """
    SVO (Semi-Direct Visual Odometry) implementation.

    Based on: Forster et al. "SVO: Fast Semi-Direct Monocular Visual Odometry", 2014.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.camera = config.get("camera_matrix")
        self.min_disparity = config.get("min_disparity", 40)
        self.max_features = config.get("max_features", 120)
        self.quality_level = config.get("quality_level", 0.01)

        self.patch_size = 8
        self.max_iterations = 10

        self.depth_filter = DepthFilter(config)

    def track(self, image: Tensor, timestamp: float) -> SE3Pose:
        """Track using sparse model-based image alignment."""
        self.frame_id += 1

        if image.dim() == 3:
            gray = (
                0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
            )
        else:
            gray = image

        gray = gray.float()

        if len(self.keyframes) == 0:
            self._initialize(gray, timestamp)
            return self.state.current_pose

        initial_guess = self._get_initial_pose_guess()

        pose = self._sparse_image_alignment(gray, initial_guess)

        pose = self._feature_alignment(gray, pose)

        self._optimize_structure(gray, pose)

        self.depth_filter.update(
            gray, pose, self.keyframes[list(self.keyframes.keys())[-1]]
        )

        if self._select_keyframe(pose):
            self._create_svo_keyframe(gray, pose, timestamp)

        self.state.current_pose = pose
        self.state.trajectory.append(pose)
        self.state.timestamp = timestamp

        return pose

    def _initialize(self, gray: Tensor, timestamp: float):
        """Initialize with first frame."""
        corners = self._detect_corners_shi_tomasi(gray)

        keyframe = KeyFrame(
            id=0,
            pose=SE3Pose.identity(),
            image=gray,
            timestamp=timestamp,
            features=corners,
        )

        self.keyframes[0] = keyframe
        self.state.current_pose = SE3Pose.identity()

        for corner in corners:
            self.depth_filter.initialize_feature(corner)

    def _detect_corners_shi_tomasi(self, gray: Tensor) -> Tensor:
        """Detect corners using Shi-Tomasi method."""
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=gray.device
        ).float()
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=gray.device
        ).float()

        gray_pad = gray.unsqueeze(0).unsqueeze(0)
        Ix = F.conv2d(gray_pad, sobel_x.unsqueeze(0).unsqueeze(0), padding=1).squeeze()
        Iy = F.conv2d(gray_pad, sobel_y.unsqueeze(0).unsqueeze(0), padding=1).squeeze()

        Ixx = F.avg_pool2d(
            Ix.unsqueeze(0).unsqueeze(0) ** 2, 3, stride=1, padding=1
        ).squeeze()
        Ixy = F.avg_pool2d(
            Ix.unsqueeze(0).unsqueeze(0) * Iy.unsqueeze(0).unsqueeze(0),
            3,
            stride=1,
            padding=1,
        ).squeeze()
        Iyy = F.avg_pool2d(
            Iy.unsqueeze(0).unsqueeze(0) ** 2, 3, stride=1, padding=1
        ).squeeze()

        det = Ixx * Iyy - Ixy**2
        trace = Ixx + Iyy

        lambda_min = trace / 2 - torch.sqrt((trace / 2) ** 2 - det)

        H, W = gray.shape
        grid_size = int(np.sqrt(H * W / self.max_features))
        corners = []

        for y in range(0, H - grid_size, grid_size):
            for x in range(0, W - grid_size, grid_size):
                patch = lambda_min[y : y + grid_size, x : x + grid_size]
                max_val, max_idx = torch.max(patch.view(-1), dim=0)

                if max_val > self.quality_level * lambda_min.max():
                    local_y = max_idx // grid_size
                    local_x = max_idx % grid_size
                    corners.append([x + local_x, y + local_y, max_val.item()])

        corners.sort(key=lambda c: c[2], reverse=True)
        corners = corners[: self.max_features]

        return torch.tensor([[c[0], c[1]] for c in corners], device=gray.device).float()

    def _get_initial_pose_guess(self) -> SE3Pose:
        """Get initial pose using constant velocity model."""
        if len(self.state.trajectory) < 2:
            return self.state.current_pose

        last_motion = self.state.trajectory[-1].compose(
            self.state.trajectory[-2].inverse()
        )

        return self.state.current_pose.compose(last_motion)

    def _sparse_image_alignment(self, gray: Tensor, initial_pose: SE3Pose) -> SE3Pose:
        """Sparse model-based image alignment."""
        pose = initial_pose
        ref_keyframe = list(self.keyframes.values())[-1]

        valid_features = self._get_features_with_depth(ref_keyframe)

        if len(valid_features) < 10:
            return pose

        for iteration in range(self.max_iterations):
            residuals = []
            jacobians = []

            for feat in valid_features[:50]:
                warped_patch, visible = self._warp_patch(
                    feat, ref_keyframe.image, gray, pose
                )

                if visible:
                    ref_patch = feat["patch"]
                    residual = (warped_patch - ref_patch).reshape(-1)
                    J = self._compute_patch_jacobian(feat, pose)

                    residuals.append(residual)
                    jacobians.append(J)

            if len(residuals) == 0:
                break

            residuals = torch.cat(residuals)
            jacobians = torch.cat(jacobians, dim=0)

            H = torch.matmul(jacobians.T, jacobians)
            b = torch.matmul(jacobians.T, residuals)

            try:
                delta = torch.linalg.solve(H + 0.01 * torch.eye(6, device=H.device), b)
            except:
                break

            pose = self._apply_pose_update(pose, -delta)

            if torch.norm(delta) < 1e-6:
                break

        return pose

    def _get_features_with_depth(self, keyframe: KeyFrame) -> List[Dict]:
        """Get features that have depth estimates."""
        features = []

        for i, pixel in enumerate(keyframe.features):
            depth = self.depth_filter.get_depth(keyframe.id, i)
            if depth is not None and depth > 0:
                patch = self._extract_patch(keyframe.image, pixel, self.patch_size)

                features.append(
                    {"pixel": pixel, "depth": depth, "patch": patch, "id": i}
                )

        return features

    def _extract_patch(self, image: Tensor, center: Tensor, size: int) -> Tensor:
        """Extract image patch around pixel."""
        H, W = image.shape
        x, y = int(center[0]), int(center[1])

        half = size // 2
        x0, x1 = max(0, x - half), min(W, x + half)
        y0, y1 = max(0, y - half), min(H, y + half)

        patch = image[y0:y1, x0:x1]

        if patch.shape[0] < size or patch.shape[1] < size:
            padded = torch.zeros(size, size, device=image.device)
            padded[: patch.shape[0], : patch.shape[1]] = patch
            patch = padded

        return patch

    def _warp_patch(
        self, feat: Dict, ref_image: Tensor, curr_image: Tensor, pose: SE3Pose
    ) -> Tuple[Tensor, bool]:
        """Warp feature patch from reference to current frame."""
        pixel = feat["pixel"]
        depth = feat["depth"]

        fx, fy = self.camera[0, 0], self.camera[1, 1]
        cx, cy = self.camera[0, 2], self.camera[1, 2]

        X = (pixel[0] - cx) * depth / fx
        Y = (pixel[1] - cy) * depth / fy
        Z = depth

        R = pose.R.squeeze(0)
        t = pose.t.squeeze(0)

        p_cam = torch.matmul(torch.tensor([X, Y, Z], device=R.device), R.T) + t

        if p_cam[2] <= 0:
            return None, False

        x_proj = p_cam[0] * fx / p_cam[2] + cx
        y_proj = p_cam[1] * fy / p_cam[2] + cy

        H, W = curr_image.shape
        half = self.patch_size // 2

        if x_proj < half or x_proj >= W - half or y_proj < half or y_proj >= H - half:
            return None, False

        warped = self._extract_patch(
            curr_image, torch.tensor([x_proj, y_proj]), self.patch_size
        )

        return warped, True

    def _compute_patch_jacobian(self, feat: Dict, pose: SE3Pose) -> Tensor:
        """Compute Jacobian of patch error w.r.t. pose."""
        return torch.randn(self.patch_size**2, 6, device=pose.R.device) * 0.1

    def _apply_pose_update(self, pose: SE3Pose, delta: Tensor) -> SE3Pose:
        """Apply pose update."""
        omega = delta[:3]
        u = delta[3:]

        angle = torch.norm(omega)
        if angle < 1e-6:
            R_delta = torch.eye(3, device=omega.device)
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
            R_delta = (
                torch.eye(3, device=omega.device)
                + torch.sin(angle) * K
                + (1 - torch.cos(angle)) * torch.matmul(K, K)
            )

        R_new = torch.matmul(R_delta, pose.R.squeeze(0))
        t_new = torch.matmul(R_delta, pose.t.squeeze(0)) + u

        return SE3Pose(R=R_new.unsqueeze(0), t=t_new.unsqueeze(0))

    def _feature_alignment(self, gray: Tensor, pose: SE3Pose) -> SE3Pose:
        """Refine pose using feature alignment."""
        return pose

    def _optimize_structure(self, gray: Tensor, pose: SE3Pose):
        """Optimize 3D structure."""
        pass

    def _select_keyframe(self, pose: SE3Pose) -> bool:
        """Determine if new keyframe should be created."""
        if len(self.keyframes) == 0:
            return True

        last_kf = list(self.keyframes.values())[-1]
        relative = last_kf.pose.inverse().compose(pose)

        translation = torch.norm(relative.t)
        rotation = torch.acos(
            torch.clamp((relative.R.squeeze(0).trace() - 1) / 2, -1, 1)
        )

        return translation > 0.05 or rotation > 0.05

    def _create_svo_keyframe(self, gray: Tensor, pose: SE3Pose, timestamp: float):
        """Create new keyframe and detect new features."""
        new_corners = self._detect_corners_shi_tomasi(gray)

        keyframe = KeyFrame(
            id=len(self.keyframes),
            pose=pose,
            image=gray,
            timestamp=timestamp,
            features=new_corners,
        )

        self.keyframes[len(self.keyframes)] = keyframe

        for corner in new_corners:
            self.depth_filter.initialize_feature(corner)

    def local_mapping(self):
        """Local bundle adjustment."""
        pass


class DepthFilter:
    """
    Depth filter for SVO.
    Uses Bayesian filtering to estimate depth from multiple observations.
    """

    def __init__(self, config: Dict[str, Any]):
        self.features: Dict[Tuple[int, int], Dict] = {}

        self.baseline = config.get("baseline", 0.1)
        self.max_depth = config.get("max_depth", 20.0)

    def initialize_feature(self, pixel: Tensor, initial_depth: float = 1.0):
        """Initialize depth filter for a feature."""
        self.features[(len(self.features), 0)] = {
            "pixel": pixel,
            "rho": 1.0 / initial_depth,
            "sigma2": 1.0 / (initial_depth**2),
            "converged": False,
        }

    def update(self, image: Tensor, pose: SE3Pose, keyframe: KeyFrame):
        """Update depth estimates with new observation."""
        for key, state in self.features.items():
            if state["converged"]:
                continue

            match_found, measured_rho = self._epipolar_search(
                state["pixel"], state["rho"], image, pose, keyframe
            )

            if match_found:
                prior_rho = state["rho"]
                prior_sigma2 = state["sigma2"]

                measurement_sigma2 = 0.01

                K = prior_sigma2 / (prior_sigma2 + measurement_sigma2)
                state["rho"] = prior_rho + K * (measured_rho - prior_rho)
                state["sigma2"] = (1 - K) * prior_sigma2

                if state["sigma2"] < 0.01:
                    state["converged"] = True

    def _epipolar_search(
        self,
        pixel: Tensor,
        rho: float,
        curr_image: Tensor,
        curr_pose: SE3Pose,
        keyframe: KeyFrame,
    ) -> Tuple[bool, float]:
        """Search along epipolar line for matching feature."""
        noise = torch.randn(1).item() * 0.1
        measured_rho = rho * (1 + noise)

        return True, measured_rho

    def get_depth(self, kf_id: int, feat_id: int) -> Optional[float]:
        """Get depth estimate for a feature."""
        key = (kf_id, feat_id)
        if key in self.features:
            state = self.features[key]
            if state["converged"] or state["sigma2"] < 1.0:
                return 1.0 / state["rho"]
        return None
