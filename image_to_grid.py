import cv2
import numpy as np
from PIL import Image

class ImageToGrid:
    def __init__(self, threshold=127, min_cell_size=5):
        """
        Initialize the image to grid converter.
        
        Args:
            threshold: Threshold for binary conversion (0-255)
            min_cell_size: Minimum size of maze cells in pixels
        """
        self.threshold = threshold
        self.min_cell_size = min_cell_size
    
    def convert(self, image_path):
        """
        Convert maze image to 2D binary grid and detect start/end points.
        
        Args:
            image_path: Path to the maze image file
            
        Returns:
            Tuple of (grid, start_pos, end_pos)
            grid: 2D list where 0=path, 1=wall
            start_pos: (row, col) tuple for start position
            end_pos: (row, col) tuple for end position
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Preprocess the image
        processed_image = self._preprocess_image(image)
        
        # Convert to binary grid
        grid = self._image_to_binary_grid(processed_image)
        
        # Clean up the grid
        grid = self._clean_grid(grid)
        
        # Detect start and end points
        start_pos, end_pos = self._detect_start_end_points(grid, image)
        
        return grid, start_pos, end_pos
    
    def _preprocess_image(self, image):
        """
        Preprocess the image for better maze detection.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding for better edge detection
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def _image_to_binary_grid(self, image):
        """
        Convert preprocessed image to binary grid.
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            2D list representing the maze grid
        """
        height, width = image.shape
        
        # Determine appropriate cell size based on image dimensions
        cell_size = max(self.min_cell_size, min(width, height) // 100)
        
        # Calculate grid dimensions
        grid_height = height // cell_size
        grid_width = width // cell_size
        
        # Create binary grid
        grid = []
        for row in range(grid_height):
            grid_row = []
            for col in range(grid_width):
                # Calculate the region in the image for this grid cell
                y1 = row * cell_size
                y2 = min((row + 1) * cell_size, height)
                x1 = col * cell_size
                x2 = min((col + 1) * cell_size, width)
                
                # Extract the cell region
                cell_region = image[y1:y2, x1:x2]
                
                # Determine if this cell is a wall or path
                # If most pixels are dark (below threshold), it's a wall
                avg_intensity = np.mean(cell_region)
                is_wall = 1 if avg_intensity < self.threshold else 0
                
                grid_row.append(is_wall)
            grid.append(grid_row)
        
        return grid
    
    def _clean_grid(self, grid):
        """
        Clean up the grid by removing isolated pixels and small artifacts.
        
        Args:
            grid: 2D list representing the maze
            
        Returns:
            Cleaned 2D grid
        """
        height, width = len(grid), len(grid[0])
        cleaned_grid = [row[:] for row in grid]  # Deep copy
        
        # Remove isolated wall pixels (walls surrounded by paths)
        for row in range(1, height - 1):
            for col in range(1, width - 1):
                if grid[row][col] == 1:  # Wall pixel
                    # Count wall neighbors
                    wall_neighbors = 0
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        if grid[row + dr][col + dc] == 1:
                            wall_neighbors += 1
                    
                    # If isolated wall (no wall neighbors), convert to path
                    if wall_neighbors == 0:
                        cleaned_grid[row][col] = 0
        
        # Remove isolated path pixels (paths surrounded by walls)
        for row in range(1, height - 1):
            for col in range(1, width - 1):
                if grid[row][col] == 0:  # Path pixel
                    # Count path neighbors
                    path_neighbors = 0
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        if grid[row + dr][col + dc] == 0:
                            path_neighbors += 1
                    
                    # If isolated path (no path neighbors), convert to wall
                    if path_neighbors == 0:
                        cleaned_grid[row][col] = 1
        
        # Ensure borders are mostly walls (typical maze property)
        self._fix_borders(cleaned_grid)
        
        return cleaned_grid
    
    def _fix_borders(self, grid):
        """
        Ensure maze borders are properly formed.
        
        Args:
            grid: 2D grid to modify in-place
        """
        height, width = len(grid), len(grid[0])
        
        # Make most border cells walls, but preserve potential entrances/exits
        border_positions = []
        
        # Top and bottom borders
        for col in range(width):
            border_positions.extend([(0, col), (height - 1, col)])
        
        # Left and right borders
        for row in range(1, height - 1):
            border_positions.extend([(row, 0), (row, width - 1)])
        
        # Count path cells on borders
        border_paths = sum(1 for row, col in border_positions if grid[row][col] == 0)
        
        # If too many border paths, keep only a few as entrances/exits
        if border_paths > 4:  # Allow maximum 4 openings
            path_borders = [(row, col) for row, col in border_positions if grid[row][col] == 0]
            
            # Keep 2-4 openings, preferably at corners or edges
            openings_to_keep = min(4, max(2, border_paths // 3))
            
            # Prioritize corner and edge positions
            priority_positions = [
                (0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1),  # Corners
                (0, width // 2), (height - 1, width // 2),  # Top/bottom centers
                (height // 2, 0), (height // 2, width - 1)   # Left/right centers
            ]
            
            # Keep high-priority positions first
            kept_openings = []
            for pos in priority_positions:
                if pos in path_borders and len(kept_openings) < openings_to_keep:
                    kept_openings.append(pos)
            
            # Fill remaining openings randomly
            remaining_paths = [pos for pos in path_borders if pos not in kept_openings]
            while len(kept_openings) < openings_to_keep and remaining_paths:
                kept_openings.append(remaining_paths.pop(0))
            
            # Close all other border paths
            for row, col in path_borders:
                if (row, col) not in kept_openings:
                    grid[row][col] = 1
    
    def _detect_start_end_points(self, grid, original_image):
        """
        Improved start/end detection with color detection priority.
        """
        height, width = len(grid), len(grid[0])
        
        # PRIORITY 1: Try color detection first (scan entire image)
        start_pos, end_pos = self._detect_colored_points(original_image, [], grid)
        
        if start_pos and end_pos:
            print(f"Successfully found colored start/end: {start_pos}, {end_pos}")
            # Ensure both positions are valid paths
            if grid[start_pos[0]][start_pos[1]] == 1:
                grid[start_pos[0]][start_pos[1]] = 0
                print(f"Converted start position {start_pos} from wall to path")
            
            if grid[end_pos[0]][end_pos[1]] == 1:
                grid[end_pos[0]][end_pos[1]] = 0
                print(f"Converted end position {end_pos} from wall to path")
            
            return start_pos, end_pos
        
        # PRIORITY 2: Find border openings as fallback
        border_openings = []
        
        # Check all border positions for openings
        for row in range(height):
            for col in range(width):
                if (row == 0 or row == height - 1 or col == 0 or col == width - 1):
                    if grid[row][col] == 0:  # Path on border
                        border_openings.append((row, col))
        
        print(f"Found {len(border_openings)} border openings: {border_openings}")
        
        if len(border_openings) >= 2:
            # Use the most distant pair of border openings
            max_distance = 0
            best_pair = (border_openings[0], border_openings[-1])
            
            for i in range(len(border_openings)):
                for j in range(i + 1, len(border_openings)):
                    pos1, pos2 = border_openings[i], border_openings[j]
                    distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                    
                    if distance > max_distance:
                        max_distance = distance
                        best_pair = (pos1, pos2)
            
            start_pos, end_pos = best_pair
        
        elif len(border_openings) == 1:
            # Only one opening found, create another
            start_pos = border_openings[0]
            end_pos = self._find_farthest_opening(start_pos, grid)
        
        else:
            # No border openings found, create default ones
            print("No border openings found, creating default start/end points")
            start_pos, end_pos = self._create_default_start_end(grid)
        
        # Ensure both positions are valid paths
        if start_pos and grid[start_pos[0]][start_pos[1]] == 1:
            grid[start_pos[0]][start_pos[1]] = 0
            print(f"Converted start position {start_pos} from wall to path")
        
        if end_pos and grid[end_pos[0]][end_pos[1]] == 1:
            grid[end_pos[0]][end_pos[1]] = 0
            print(f"Converted end position {end_pos} from wall to path")
        
        print(f"Final start/end positions: {start_pos}, {end_pos}")
        return start_pos, end_pos
    
    def _detect_colored_points(self, image, border_openings, grid):
        """
        Improved colored point detection for start (green) and end (red) points.
        """
        height, width = len(grid), len(grid[0])
        img_height, img_width = image.shape[:2]
        
        # Calculate scaling factors
        scale_y = img_height / height
        scale_x = img_width / width
        
        start_pos = None
        end_pos = None
        
        # Scan the entire image for colored markers, not just border openings
        print("Scanning image for colored start/end points...")
        
        # Check every grid cell for colored markers
        for grid_row in range(height):
            for grid_col in range(width):
                # Convert grid coordinates to image coordinates
                img_y = int(grid_row * scale_y + scale_y // 2)
                img_x = int(grid_col * scale_x + scale_x // 2)
                
                # Sample color in a region around the point
                region_size = max(3, int(min(scale_x, scale_y) // 3))
                y1 = max(0, img_y - region_size)
                y2 = min(img_height, img_y + region_size)
                x1 = max(0, img_x - region_size)
                x2 = min(img_width, img_x + region_size)
                
                if y2 > y1 and x2 > x1:
                    region = image[y1:y2, x1:x2]
                    avg_color = np.mean(region, axis=(0, 1))  # Average BGR color
                    
                    # Check if it's predominantly green (start point)
                    if self._is_green(avg_color):
                        start_pos = (grid_row, grid_col)
                        print(f"Found GREEN start point at grid {start_pos}, image coords ({img_x}, {img_y})")
                    
                    # Check if it's predominantly red (end point)
                    elif self._is_red(avg_color):
                        end_pos = (grid_row, grid_col)
                        print(f"Found RED end point at grid {end_pos}, image coords ({img_x}, {img_y})")
        
        return start_pos, end_pos
    
    def _is_green(self, bgr_color):
        """Check if color is predominantly green with better thresholds."""
        b, g, r = bgr_color
        return g > r + 20 and g > b + 20 and g > 80
    
    def _is_red(self, bgr_color):
        """Check if color is predominantly red with better thresholds."""
        b, g, r = bgr_color
        return r > g + 20 and r > b + 20 and r > 80
    
    def _heuristic_start_end_detection(self, border_openings, grid):
        """
        Use heuristics to determine start and end points.
        
        Args:
            border_openings: List of border opening positions
            grid: Binary grid
            
        Returns:
            Tuple of (start_pos, end_pos)
        """
        if len(border_openings) == 0:
            return self._create_default_start_end(grid)
        
        if len(border_openings) == 1:
            # Only one opening, create another one
            start_pos = border_openings[0]
            end_pos = self._find_farthest_opening(start_pos, grid)
            return start_pos, end_pos
        
        # Multiple openings - use distance heuristic
        max_distance = 0
        best_pair = (border_openings[0], border_openings[-1])
        
        for i in range(len(border_openings)):
            for j in range(i + 1, len(border_openings)):
                pos1, pos2 = border_openings[i], border_openings[j]
                distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])  # Manhattan distance
                
                if distance > max_distance:
                    max_distance = distance
                    best_pair = (pos1, pos2)
        
        # Assign start as top-left-most, end as bottom-right-most
        pos1, pos2 = best_pair
        if pos1[0] + pos1[1] < pos2[0] + pos2[1]:
            return pos1, pos2
        else:
            return pos2, pos1
    
    def _create_default_start_end(self, grid):
        """
        Create default start and end points if none detected.
        
        Args:
            grid: Binary grid
            
        Returns:
            Tuple of (start_pos, end_pos)
        """
        height, width = len(grid), len(grid[0])
        
        # Find suitable positions near corners
        start_candidates = [(0, 1), (1, 0), (0, 0)]
        end_candidates = [(height-1, width-2), (height-2, width-1), (height-1, width-1)]
        
        # Find valid start position
        start_pos = None
        for pos in start_candidates:
            row, col = pos
            if 0 <= row < height and 0 <= col < width:
                grid[row][col] = 0  # Ensure it's a path
                start_pos = pos
                break
        
        # Find valid end position
        end_pos = None
        for pos in end_candidates:
            row, col = pos
            if 0 <= row < height and 0 <= col < width:
                grid[row][col] = 0  # Ensure it's a path
                end_pos = pos
                break
        
        return start_pos or (0, 0), end_pos or (height-1, width-1)
    
    def _find_farthest_opening(self, start_pos, grid):
        """
        Find or create the position farthest from start_pos.
        
        Args:
            start_pos: Starting position
            grid: Binary grid
            
        Returns:
            Position farthest from start
        """
        height, width = len(grid), len(grid[0])
        
        # Try corners first
        corners = [(0, 0), (0, width-1), (height-1, 0), (height-1, width-1)]
        
        max_distance = 0
        farthest_pos = corners[-1]  # Default to bottom-right
        
        for corner in corners:
            distance = abs(start_pos[0] - corner[0]) + abs(start_pos[1] - corner[1])
            if distance > max_distance:
                max_distance = distance
                farthest_pos = corner
        
        # Ensure the position is a path
        row, col = farthest_pos
        grid[row][col] = 0
        
        return farthest_pos
    
    def get_grid_info(self, grid):
        """
        Get information about the converted grid.
        
        Args:
            grid: 2D binary grid
            
        Returns:
            Dictionary with grid statistics
        """
        if not grid or not grid[0]:
            return {}
        
        height, width = len(grid), len(grid[0])
        total_cells = height * width
        
        # Count walls and paths
        wall_count = sum(row.count(1) for row in grid)
        path_count = total_cells - wall_count
        
        # Calculate connectivity (number of connected path components)
        connected_components = self._count_connected_components(grid)
        
        return {
            'dimensions': (height, width),
            'total_cells': total_cells,
            'wall_cells': wall_count,
            'path_cells': path_count,
            'wall_percentage': round((wall_count / total_cells) * 100, 2),
            'path_percentage': round((path_count / total_cells) * 100, 2),
            'connected_components': connected_components
        }
    
    def _count_connected_components(self, grid):
        """
        Count the number of connected path components using flood fill.
        
        Args:
            grid: 2D binary grid
            
        Returns:
            Number of connected components
        """
        if not grid or not grid[0]:
            return 0
        
        height, width = len(grid), len(grid[0])
        visited = [[False] * width for _ in range(height)]
        components = 0
        
        def flood_fill(start_row, start_col):
            """Flood fill to mark connected path cells."""
            stack = [(start_row, start_col)]
            
            while stack:
                row, col = stack.pop()
                
                if (row < 0 or row >= height or col < 0 or col >= width or
                    visited[row][col] or grid[row][col] == 1):
                    continue
                
                visited[row][col] = True
                
                # Add neighbors to stack
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((row + dr, col + dc))
        
        # Find all connected components
        for row in range(height):
            for col in range(width):
                if grid[row][col] == 0 and not visited[row][col]:
                    flood_fill(row, col)
                    components += 1
        
        return components
    
    def validate_maze(self, grid, start_pos, end_pos):
        """
        Validate that the maze is solvable.
        
        Args:
            grid: 2D binary grid
            start_pos: Start position tuple
            end_pos: End position tuple
            
        Returns:
            Boolean indicating if maze is valid and solvable
        """
        if not grid or not grid[0]:
            return False
        
        height, width = len(grid), len(grid[0])
        
        # Check if start and end positions are valid
        if not start_pos or not end_pos:
            return False
        
        start_row, start_col = start_pos
        end_row, end_col = end_pos
        
        # Check bounds
        if (start_row < 0 or start_row >= height or start_col < 0 or start_col >= width or
            end_row < 0 or end_row >= height or end_col < 0 or end_col >= width):
            return False
        
        # Check if start and end are on paths
        if grid[start_row][start_col] == 1 or grid[end_row][end_col] == 1:
            return False
        
        # Check if there's a path from start to end using BFS
        return self._is_reachable(grid, start_pos, end_pos)
    
    def _is_reachable(self, grid, start_pos, end_pos):
        """
        Check if end position is reachable from start position.
        
        Args:
            grid: 2D binary grid
            start_pos: Start position tuple
            end_pos: End position tuple
            
        Returns:
            Boolean indicating reachability
        """
        if start_pos == end_pos:
            return True
        
        height, width = len(grid), len(grid[0])
        visited = [[False] * width for _ in range(height)]
        queue = [start_pos]
        visited[start_pos[0]][start_pos[1]] = True
        
        while queue:
            row, col = queue.pop(0)
            
            # Check all four directions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc
                
                # Check bounds
                if (new_row < 0 or new_row >= height or 
                    new_col < 0 or new_col >= width):
                    continue
                
                # Skip if wall or already visited
                if grid[new_row][new_col] == 1 or visited[new_row][new_col]:
                    continue
                
                # Check if we reached the end
                if (new_row, new_col) == end_pos:
                    return True
                
                # Mark as visited and add to queue
                visited[new_row][new_col] = True
                queue.append((new_row, new_col))
        
        return False
    
    def save_debug_image(self, grid, start_pos, end_pos, output_path):
        """
        Save a debug image showing the converted grid with start/end points.
        
        Args:
            grid: 2D binary grid
            start_pos: Start position tuple
            end_pos: End position tuple
            output_path: Path to save the debug image
        """
        if not grid or not grid[0]:
            return
        
        height, width = len(grid), len(grid[0])
        cell_size = 20
        
        # Create image
        img_width = width * cell_size
        img_height = height * cell_size
        
        # Create RGB image (white background)
        debug_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
        
        # Draw maze
        for row in range(height):
            for col in range(width):
                if grid[row][col] == 1:  # Wall
                    y1, y2 = row * cell_size, (row + 1) * cell_size
                    x1, x2 = col * cell_size, (col + 1) * cell_size
                    debug_img[y1:y2, x1:x2] = [0, 0, 0]  # Black walls
        
        # Draw start point (green circle)
        if start_pos:
            center_x = start_pos[1] * cell_size + cell_size // 2
            center_y = start_pos[0] * cell_size + cell_size // 2
            cv2.circle(debug_img, (center_x, center_y), cell_size // 3, (0, 255, 0), -1)
        
        # Draw end point (red circle)
        if end_pos:
            center_x = end_pos[1] * cell_size + cell_size // 2
            center_y = end_pos[0] * cell_size + cell_size // 2
            cv2.circle(debug_img, (center_x, center_y), cell_size // 3, (0, 0, 255), -1)
        
        # Save image
        cv2.imwrite(output_path, debug_img)
    
    def auto_detect_cell_size(self, image):
        """
        Automatically detect the optimal cell size for grid conversion.
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Optimal cell size in pixels
        """
        height, width = image.shape
        
        # Use edge detection to find typical wall thickness
        edges = cv2.Canny(image, 50, 150)
        
        # Find contours to analyze structure
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return max(self.min_cell_size, min(width, height) // 50)
        
        # Analyze contour sizes to estimate cell size
        contour_areas = [cv2.contourArea(contour) for contour in contours]
        avg_contour_area = np.mean(contour_areas) if contour_areas else 0
        
        # Estimate cell size based on image dimensions and contour analysis
        estimated_cell_size = int(np.sqrt(avg_contour_area) / 2) if avg_contour_area > 0 else min(width, height) // 50
        
        # Ensure minimum cell size
        cell_size = max(self.min_cell_size, estimated_cell_size)
        
        # Ensure reasonable maximum based on image size
        max_cell_size = min(width, height) // 10
        cell_size = min(cell_size, max_cell_size)
        
        return cell_size
    
    def enhance_maze_image(self, image_path, output_path=None):
        """
        Enhance maze image for better processing.
        
        Args:
            image_path: Path to input image
            output_path: Path to save enhanced image (optional)
            
        Returns:
            Enhanced image array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Apply sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Apply binary thresholding
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up with morphological operations
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        final = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Save enhanced image if path provided
        if output_path:
            cv2.imwrite(output_path, final)
        
        return final