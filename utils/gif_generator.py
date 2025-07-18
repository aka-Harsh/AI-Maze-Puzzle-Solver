import cv2
import numpy as np
from PIL import Image, ImageDraw
import imageio
import os

class GifGenerator:
    """
    Generate animated GIFs showing pathfinding algorithm execution.
    """
    
    def __init__(self, frame_duration=0.1, cell_size=20):
        self.frame_duration = frame_duration  # Duration per frame in seconds
        self.cell_size = cell_size
        self.colors = {
            'wall': (0, 0, 0),        # Black
            'path': (255, 255, 255),   # White
            'start': (0, 255, 0),      # Green
            'end': (255, 0, 0),        # Red
            'visited': (128, 128, 128), # Gray
            'current': (0, 0, 255),    # Blue
            'final_path': (255, 255, 0) # Yellow
        }
    
    def create_pathfinding_animation(self, original_image, path, visited, start, end, output_path):
        """
        Create animated GIF showing pathfinding process.
        
        Args:
            original_image: Original maze image
            path: Final path as list of coordinates
            visited: All visited nodes in order
            start: Start position
            end: End position
            output_path: Path to save the GIF
        """
        try:
            # Convert image to grid dimensions
            grid = self._image_to_simple_grid(original_image)
            frames = []
            
            # Create base frame
            base_frame = self._create_maze_frame(grid, start, end)
            
            # Add frames for exploration
            visited_so_far = set()
            for i, node in enumerate(visited):
                if i % 3 == 0:  # Skip some frames for performance
                    visited_so_far.add(node)
                    frame = base_frame.copy()
                    
                    # Draw visited nodes
                    for v_node in visited_so_far:
                        if v_node != start and v_node != end:
                            self._draw_cell(frame, v_node, self.colors['visited'])
                    
                    # Highlight current node
                    if node != start and node != end:
                        self._draw_cell(frame, node, self.colors['current'])
                    
                    frames.append(frame)
            
            # Add frames showing final path
            path_so_far = []
            for node in path:
                if node != start and node != end:
                    path_so_far.append(node)
                    frame = base_frame.copy()
                    
                    # Draw all visited (faded)
                    for v_node in visited:
                        if v_node != start and v_node != end and v_node not in path_so_far:
                            self._draw_cell(frame, v_node, (200, 200, 200))
                    
                    # Draw path so far
                    for p_node in path_so_far:
                        self._draw_cell(frame, p_node, self.colors['final_path'])
                    
                    frames.append(frame)
            
            # Hold final frame
            final_frame = frames[-1] if frames else base_frame
            for _ in range(10):  # Hold for 1 second
                frames.append(final_frame)
            
            # Save as GIF
            if frames:
                # Convert frames to PIL Images
                pil_frames = []
                for frame in frames:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frames.append(Image.fromarray(rgb_frame))
                
                # Save GIF
                pil_frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=int(self.frame_duration * 1000),  # Convert to milliseconds
                    loop=0
                )
            
        except Exception as e:
            print(f"Error creating animation: {e}")
            # Create a simple static image as fallback
            self._create_static_fallback(original_image, path, start, end, output_path)
    
    def _image_to_simple_grid(self, image):
        """Convert image to simple binary grid for animation."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        height, width = gray.shape
        # Downsample for animation performance
        cell_size = max(10, min(width, height) // 50)
        
        grid_height = height // cell_size
        grid_width = width // cell_size
        
        grid = []
        for row in range(grid_height):
            grid_row = []
            for col in range(grid_width):
                y1, y2 = row * cell_size, min((row + 1) * cell_size, height)
                x1, x2 = col * cell_size, min((col + 1) * cell_size, width)
                
                cell_region = gray[y1:y2, x1:x2]
                avg_intensity = np.mean(cell_region)
                is_wall = 1 if avg_intensity < 127 else 0
                grid_row.append(is_wall)
            grid.append(grid_row)
        
        return grid
    
    def _create_maze_frame(self, grid, start, end):
        """Create base maze frame."""
        height, width = len(grid), len(grid[0])
        frame_height = height * self.cell_size
        frame_width = width * self.cell_size
        
        # Create frame
        frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
        
        # Draw maze
        for row in range(height):
            for col in range(width):
                if grid[row][col] == 1:  # Wall
                    self._draw_cell(frame, (row, col), self.colors['wall'])
        
        # Draw start and end
        self._draw_cell(frame, start, self.colors['start'])
        self._draw_cell(frame, end, self.colors['end'])
        
        return frame
    
    def _draw_cell(self, frame, pos, color):
        """Draw a colored cell on the frame."""
        row, col = pos
        y1 = row * self.cell_size
        y2 = (row + 1) * self.cell_size
        x1 = col * self.cell_size
        x2 = (col + 1) * self.cell_size
        
        # Ensure we don't go out of bounds
        frame_height, frame_width = frame.shape[:2]
        y1, y2 = max(0, y1), min(frame_height, y2)
        x1, x2 = max(0, x1), min(frame_width, x2)
        
        if y2 > y1 and x2 > x1:
            frame[y1:y2, x1:x2] = color
    
    def _create_static_fallback(self, original_image, path, start, end, output_path):
        """Create static image as fallback if GIF creation fails."""
        try:
            # Create simple path visualization
            result_image = original_image.copy()
            
            # Draw path
            if len(path) > 1:
                height, width = len(path), len(path[0]) if path else 1
                img_height, img_width = result_image.shape[:2]
                
                scale_y = img_height / height if height > 0 else 1
                scale_x = img_width / width if width > 0 else 1
                
                # Draw path lines
                for i in range(len(path) - 1):
                    y1, x1 = path[i]
                    y2, x2 = path[i + 1]
                    
                    point1 = (int(x1 * scale_x), int(y1 * scale_y))
                    point2 = (int(x2 * scale_x), int(y2 * scale_y))
                    
                    cv2.line(result_image, point1, point2, (0, 255, 255), 3)
            
            # Convert to PIL and save as GIF (single frame)
            rgb_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            pil_image.save(output_path)
            
        except Exception as e:
            print(f"Error creating fallback image: {e}")
    
    def create_comparison_animation(self, original_image, algorithm_results, output_path):
        """
        Create comparison animation showing multiple algorithms.
        
        Args:
            original_image: Original maze image
            algorithm_results: Dict of {algorithm_name: (path, visited)}
            output_path: Path to save comparison GIF
        """
        try:
            grid = self._image_to_simple_grid(original_image)
            frames = []
            
            # Create frames for each algorithm
            for algo_name, (path, visited) in algorithm_results.items():
                # Create algorithm frame
                frame = self._create_maze_frame(grid, visited[0] if visited else (0, 0), 
                                              path[-1] if path else (0, 0))
                
                # Add algorithm name
                cv2.putText(frame, algo_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (255, 255, 255), 2)
                
                # Draw visited nodes
                for node in visited:
                    if node not in [visited[0], path[-1]] if visited and path else []:
                        self._draw_cell(frame, node, self.colors['visited'])
                
                # Draw final path
                for node in path:
                    if node not in [visited[0], path[-1]] if visited and path else []:
                        self._draw_cell(frame, node, self.colors['final_path'])
                
                # Hold frame for 2 seconds
                for _ in range(20):
                    frames.append(frame)
            
            # Save comparison GIF
            if frames:
                pil_frames = []
                for frame in frames:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frames.append(Image.fromarray(rgb_frame))
                
                pil_frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=100,
                    loop=0
                )
                
        except Exception as e:
            print(f"Error creating comparison animation: {e}")
    
    def create_step_by_step_images(self, original_image, visited, path, start, end, output_dir):
        """
        Create individual step images for manual animation control.
        
        Args:
            original_image: Original maze image
            visited: Visited nodes in order
            path: Final path
            start: Start position
            end: End position
            output_dir: Directory to save step images
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            grid = self._image_to_simple_grid(original_image)
            
            visited_so_far = set()
            step = 0
            
            # Create exploration steps
            for i, node in enumerate(visited):
                if i % 5 == 0:  # Every 5th step
                    visited_so_far.add(node)
                    frame = self._create_maze_frame(grid, start, end)
                    
                    # Draw visited
                    for v_node in visited_so_far:
                        if v_node != start and v_node != end:
                            self._draw_cell(frame, v_node, self.colors['visited'])
                    
                    # Highlight current
                    if node != start and node != end:
                        self._draw_cell(frame, node, self.colors['current'])
                    
                    # Save frame
                    cv2.imwrite(f"{output_dir}/step_{step:04d}.png", frame)
                    step += 1
            
            # Create path reconstruction steps
            path_so_far = []
            for node in path:
                if node != start and node != end:
                    path_so_far.append(node)
                    frame = self._create_maze_frame(grid, start, end)
                    
                    # Draw faded visited
                    for v_node in visited:
                        if v_node != start and v_node != end and v_node not in path_so_far:
                            self._draw_cell(frame, v_node, (200, 200, 200))
                    
                    # Draw path
                    for p_node in path_so_far:
                        self._draw_cell(frame, p_node, self.colors['final_path'])
                    
                    cv2.imwrite(f"{output_dir}/path_{step:04d}.png", frame)
                    step += 1
                    
        except Exception as e:
            print(f"Error creating step images: {e}")
    
    def create_algorithm_showcase(self, original_image, results, output_path):
        """
        Create showcase animation highlighting different algorithm characteristics.
        """
        try:
            grid = self._image_to_simple_grid(original_image)
            frames = []
            
            # Title frame
            title_frame = np.ones((400, 600, 3), dtype=np.uint8) * 255
            cv2.putText(title_frame, "Pathfinding Algorithms", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
            cv2.putText(title_frame, "Comparison", (200, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            
            for _ in range(30):  # 3 second title
                frames.append(title_frame)
            
            # Algorithm comparison frames
            for algo_name, (path, visited, stats) in results.items():
                # Create info frame
                info_frame = self._create_maze_frame(grid, 
                                                   visited[0] if visited else (0, 0),
                                                   path[-1] if path else (0, 0))
                
                # Add algorithm info
                cv2.putText(info_frame, algo_name, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                if 'time' in stats:
                    cv2.putText(info_frame, f"Time: {stats['time']:.2f}ms", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if 'path_length' in stats:
                    cv2.putText(info_frame, f"Path Length: {stats['path_length']}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Animate exploration
                visited_progressive = set()
                for node in visited[::max(1, len(visited)//20)]:  # Sample nodes
                    visited_progressive.add(node)
                    frame = info_frame.copy()
                    
                    for v_node in visited_progressive:
                        if v_node != visited[0] and v_node != path[-1]:
                            self._draw_cell(frame, v_node, self.colors['visited'])
                    
                    frames.append(frame)
                
                # Show final path
                final_frame = info_frame.copy()
                for v_node in visited:
                    if v_node != visited[0] and v_node != path[-1]:
                        self._draw_cell(final_frame, v_node, (200, 200, 200))
                
                for p_node in path:
                    if p_node != visited[0] and p_node != path[-1]:
                        self._draw_cell(final_frame, p_node, self.colors['final_path'])
                
                for _ in range(20):  # Hold final result
                    frames.append(final_frame)
            
            # Save showcase
            if frames:
                pil_frames = []
                for frame in frames:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frames.append(Image.fromarray(rgb_frame))
                
                pil_frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=100,
                    loop=0
                )
                
        except Exception as e:
            print(f"Error creating showcase: {e}")
            
    def set_animation_speed(self, speed_multiplier):
        """
        Adjust animation speed.
        
        Args:
            speed_multiplier: Float where 1.0 is normal speed, 2.0 is double speed, etc.
        """
        self.frame_duration = max(0.05, 0.1 / speed_multiplier)
    
    def set_colors(self, color_scheme):
        """
        Set custom color scheme.
        
        Args:
            color_scheme: Dict with color mappings
        """
        self.colors.update(color_scheme)