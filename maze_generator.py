import random
import numpy as np
import cv2
from PIL import Image, ImageDraw

class MazeGenerator:
    def __init__(self, width=21, height=21):
        """
        Initialize maze generator with specified dimensions.
        Width and height should be odd numbers for proper maze generation.
        """
        self.width = width if width % 2 == 1 else width + 1
        self.height = height if height % 2 == 1 else height + 1
        self.maze = None
        
    def generate(self):
        """
        Generate a maze using recursive backtracking algorithm.
        Returns a 2D grid where 0 = path and 1 = wall.
        """
        # Initialize maze with all walls
        self.maze = [[1 for _ in range(self.width)] for _ in range(self.height)]
        
        # Start from position (1, 1)
        start_x, start_y = 1, 1
        self.maze[start_y][start_x] = 0
        
        # Use recursive backtracking to generate maze
        self._recursive_backtrack(start_x, start_y)
        
        # Ensure start and end points are accessible
        self._create_entrance_exit()
        
        return self.maze
    
    def _recursive_backtrack(self, x, y):
        """
        Recursive backtracking algorithm to carve out maze paths.
        """
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]  # Right, Down, Left, Up
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check if the new position is within bounds and is a wall
            if (0 < nx < self.width - 1 and 
                0 < ny < self.height - 1 and 
                self.maze[ny][nx] == 1):
                
                # Carve the path
                self.maze[ny][nx] = 0
                self.maze[y + dy // 2][x + dx // 2] = 0
                
                # Recursively continue from the new position
                self._recursive_backtrack(nx, ny)
    
    def _create_entrance_exit(self):
        """
        Create entrance and exit points on the maze borders.
        """
        # Create entrance at top-left area
        for i in range(1, min(3, self.height - 1)):
            if self.maze[i][1] == 0:
                self.maze[0][1] = 0  # Top entrance
                break
        else:
            # Force create entrance if none found
            self.maze[0][1] = 0
            self.maze[1][1] = 0
        
        # Create exit at bottom-right area
        for i in range(max(self.height - 3, 1), self.height - 1):
            if self.maze[i][self.width - 2] == 0:
                self.maze[self.height - 1][self.width - 2] = 0  # Bottom exit
                break
        else:
            # Force create exit if none found
            self.maze[self.height - 1][self.width - 2] = 0
            self.maze[self.height - 2][self.width - 2] = 0
    
    def save_image(self, maze_grid, filepath, cell_size=20):
        """
        Save the maze as an image file.
        
        Args:
            maze_grid: 2D list representing the maze
            filepath: Path to save the image
            cell_size: Size of each cell in pixels
        """
        if not maze_grid:
            raise ValueError("No maze to save")
        
        height, width = len(maze_grid), len(maze_grid[0])
        img_width = width * cell_size
        img_height = height * cell_size
        
        # Create image using PIL
        img = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(img)
        
        for y in range(height):
            for x in range(width):
                if maze_grid[y][x] == 1:  # Wall
                    x1, y1 = x * cell_size, y * cell_size
                    x2, y2 = x1 + cell_size, y1 + cell_size
                    draw.rectangle([x1, y1, x2, y2], fill='black')
        
        # Mark start and end points
        start_pos = self.find_start_end(maze_grid)
        if start_pos:
            start, end = start_pos
            
            # Draw start point (green circle)
            start_x, start_y = start[1] * cell_size, start[0] * cell_size
            start_center = (start_x + cell_size // 2, start_y + cell_size // 2)
            radius = cell_size // 3
            draw.ellipse([
                start_center[0] - radius, start_center[1] - radius,
                start_center[0] + radius, start_center[1] + radius
            ], fill='green')
            
            # Draw end point (red circle)
            end_x, end_y = end[1] * cell_size, end[0] * cell_size
            end_center = (end_x + cell_size // 2, end_y + cell_size // 2)
            draw.ellipse([
                end_center[0] - radius, end_center[1] - radius,
                end_center[0] + radius, end_center[1] + radius
            ], fill='red')
        
        # Save image
        img.save(filepath)
    
    def find_start_end(self, maze_grid):
        """
        Find start and end points in the maze.
        
        Returns:
            Tuple of (start, end) coordinates or None if not found
        """
        height, width = len(maze_grid), len(maze_grid[0])
        start = None
        end = None
        
        # Look for entrance (top border)
        for x in range(width):
            if maze_grid[0][x] == 0:
                start = (0, x)
                break
        
        # Look for exit (bottom border)
        for x in range(width):
            if maze_grid[height - 1][x] == 0:
                end = (height - 1, x)
                break
        
        # If not found on top/bottom, check left/right borders
        if not start:
            for y in range(height):
                if maze_grid[y][0] == 0:
                    start = (y, 0)
                    break
        
        if not end:
            for y in range(height):
                if maze_grid[y][width - 1] == 0:
                    end = (y, width - 1)
                    break
        
        # If still not found, use corners
        if not start:
            start = (1, 1)  # Default start
        if not end:
            end = (height - 2, width - 2)  # Default end
        
        return start, end
    
    def add_complexity(self, maze_grid, complexity_level=0.1):
        """
        Add additional complexity to the maze by creating more branching paths.
        
        Args:
            maze_grid: 2D list representing the maze
            complexity_level: Float between 0-1 representing additional path density
        """
        height, width = len(maze_grid), len(maze_grid[0])
        
        # Randomly remove some walls to create more paths
        wall_positions = []
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if maze_grid[y][x] == 1:
                    # Check if removing this wall would create a valid path
                    if self._can_remove_wall(maze_grid, x, y):
                        wall_positions.append((x, y))
        
        # Remove a percentage of walls based on complexity level
        num_to_remove = int(len(wall_positions) * complexity_level)
        walls_to_remove = random.sample(wall_positions, min(num_to_remove, len(wall_positions)))
        
        for x, y in walls_to_remove:
            maze_grid[y][x] = 0
        
        return maze_grid
    
    def _can_remove_wall(self, maze_grid, x, y):
        """
        Check if removing a wall at position (x, y) would create a valid path.
        A wall can be removed if it connects exactly two path cells.
        """
        height, width = len(maze_grid), len(maze_grid[0])
        path_neighbors = 0
        
        # Check all four directions
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if maze_grid[ny][nx] == 0:
                    path_neighbors += 1
        
        # Only remove wall if it connects exactly 2 path cells
        return path_neighbors == 2
    
    def generate_with_theme(self, theme="classic"):
        """
        Generate themed mazes with different characteristics.
        
        Args:
            theme: String indicating maze theme ('classic', 'sparse', 'dense', 'branching')
        """
        base_maze = self.generate()
        
        if theme == "sparse":
            # Create a maze with fewer walls (more open)
            base_maze = self.add_complexity(base_maze, complexity_level=0.3)
        elif theme == "dense":
            # Create a maze with more walls (more challenging)
            base_maze = self._add_dead_ends(base_maze)
        elif theme == "branching":
            # Create a maze with many branching paths
            base_maze = self.add_complexity(base_maze, complexity_level=0.2)
            base_maze = self._add_loops(base_maze)
        
        return base_maze
    
    def _add_dead_ends(self, maze_grid):
        """
        Add dead ends to make the maze more challenging.
        """
        height, width = len(maze_grid), len(maze_grid[0])
        
        for y in range(2, height - 2, 2):
            for x in range(2, width - 2, 2):
                if maze_grid[y][x] == 0 and random.random() < 0.3:
                    # Create a dead end by adding walls around the cell
                    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                    random.shuffle(directions)
                    
                    walls_added = 0
                    for dx, dy in directions:
                        if walls_added >= 3:  # Leave one opening
                            break
                        nx, ny = x + dx, y + dy
                        if (0 < nx < width - 1 and 0 < ny < height - 1 and 
                            maze_grid[ny][nx] == 0):
                            maze_grid[ny][nx] = 1
                            walls_added += 1
        
        return maze_grid
    
    def _add_loops(self, maze_grid):
        """
        Add loops to create multiple paths to the destination.
        """
        height, width = len(maze_grid), len(maze_grid[0])
        
        # Randomly connect some separate path sections
        for _ in range(max(1, (height * width) // 100)):
            x = random.randint(1, width - 2)
            y = random.randint(1, height - 2)
            
            if maze_grid[y][x] == 1:
                # Check if this wall separates two path areas
                adjacent_paths = []
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < width and 0 <= ny < height and 
                        maze_grid[ny][nx] == 0):
                        adjacent_paths.append((nx, ny))
                
                # If this wall separates paths, remove it to create a loop
                if len(adjacent_paths) >= 2:
                    maze_grid[y][x] = 0
        
        return maze_grid
    
    def get_maze_stats(self, maze_grid):
        """
        Calculate statistics about the generated maze.
        
        Returns:
            Dictionary containing maze statistics
        """
        if not maze_grid:
            return {}
        
        height, width = len(maze_grid), len(maze_grid[0])
        total_cells = height * width
        
        # Count walls and paths
        wall_count = sum(row.count(1) for row in maze_grid)
        path_count = total_cells - wall_count
        
        # Calculate wall density
        wall_density = wall_count / total_cells
        
        # Count dead ends
        dead_ends = 0
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if maze_grid[y][x] == 0:  # Path cell
                    wall_neighbors = 0
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if maze_grid[ny][nx] == 1:
                            wall_neighbors += 1
                    if wall_neighbors == 3:  # Dead end has 3 wall neighbors
                        dead_ends += 1
        
        # Count branching points
        branches = 0
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if maze_grid[y][x] == 0:  # Path cell
                    path_neighbors = 0
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if maze_grid[ny][nx] == 0:
                            path_neighbors += 1
                    if path_neighbors > 2:  # Branching point
                        branches += 1
        
        return {
            'dimensions': f"{width}x{height}",
            'total_cells': total_cells,
            'wall_count': wall_count,
            'path_count': path_count,
            'wall_density': round(wall_density, 3),
            'dead_ends': dead_ends,
            'branches': branches,
            'complexity_score': round((dead_ends + branches) / path_count, 3) if path_count > 0 else 0
        }