from flask import Flask, request, jsonify, render_template, send_file
import os
import cv2
import numpy as np
import time
import shutil
from werkzeug.utils import secure_filename
import json
from datetime import datetime

# Import our custom modules
from maze_generator import MazeGenerator
from image_to_grid import ImageToGrid
from utils.gif_generator import GifGenerator
from path_algorithms.bfs import BFS
from path_algorithms.dfs import DFS
from path_algorithms.dijkstra import Dijkstra
from path_algorithms.astar import AStar
from path_algorithms.greedy import GreedyBestFirst
from path_algorithms.bidirectional_bfs import BidirectionalBFS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/paths', exist_ok=True)

# Global variables
current_path_number = 1
algorithms = {
    'bfs': BFS(),
    'dfs': DFS(),
    'dijkstra': Dijkstra(),
    'astar': AStar(),
    'greedy': GreedyBestFirst(),
    'bidirectional_bfs': BidirectionalBFS()
}

def get_next_path_folder():
    """Get the next available path folder"""
    global current_path_number
    while os.path.exists(f'static/paths/Path{current_path_number}'):
        current_path_number += 1
    path_folder = f'static/paths/Path{current_path_number}'
    os.makedirs(path_folder, exist_ok=True)
    return path_folder

def calculate_maze_difficulty(grid):
    """Calculate maze difficulty based on various factors"""
    height, width = len(grid), len(grid[0])
    total_cells = height * width
    
    # Calculate wall density
    wall_count = sum(row.count(1) for row in grid)
    wall_density = wall_count / total_cells
    
    # Calculate path complexity (branching factor)
    path_cells = []
    for i in range(height):
        for j in range(width):
            if grid[i][j] == 0:  # Path cell
                path_cells.append((i, j))
    
    # Count decision points (cells with more than 2 neighbors)
    decision_points = 0
    for i, j in path_cells:
        neighbors = 0
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < height and 0 <= nj < width and grid[ni][nj] == 0:
                neighbors += 1
        if neighbors > 2:
            decision_points += 1
    
    # Calculate difficulty score
    branching_factor = decision_points / len(path_cells) if path_cells else 0
    difficulty_score = wall_density * 0.4 + branching_factor * 0.6
    
    if difficulty_score < 0.3:
        return "Easy"
    elif difficulty_score < 0.6:
        return "Medium"
    else:
        return "Hard"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_maze():
    try:
        if 'maze_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['maze_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the maze
            result = process_maze(filepath)
            return jsonify(result)
        
        return jsonify({'error': 'Invalid file format'}), 400
    
    except Exception as e:
        print(f"Error in upload_maze: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/generate_maze', methods=['POST'])
def generate_maze():
    try:
        data = request.get_json()
        width = data.get('width', 21) if data else 21
        height = data.get('height', 21) if data else 21
        
        # Generate maze
        generator = MazeGenerator(width, height)
        maze_grid = generator.generate()
        
        # Save maze image
        filename = f'generated_maze_{int(time.time())}.png'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        generator.save_image(maze_grid, filepath)
        
        # Process the maze
        result = process_maze(filepath)
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in generate_maze: {str(e)}")
        return jsonify({'error': f'Generation failed: {str(e)}'}), 500

def debug_grid_connectivity(grid, start, end):
    """Debug function to check if start and end are connected"""
    if not grid or not start or not end:
        return False, "Invalid grid or positions"
    
    rows, cols = len(grid), len(grid[0])
    
    # Check if start and end positions are valid paths
    start_valid = (0 <= start[0] < rows and 0 <= start[1] < cols and grid[start[0]][start[1]] == 0)
    end_valid = (0 <= end[0] < rows and 0 <= end[1] < cols and grid[end[0]][end[1]] == 0)
    
    if not start_valid:
        return False, f"Start position {start} is not a valid path (value: {grid[start[0]][start[1]] if 0 <= start[0] < rows and 0 <= start[1] < cols else 'out of bounds'})"
    
    if not end_valid:
        return False, f"End position {end} is not a valid path (value: {grid[end[0]][end[1]] if 0 <= end[0] < rows and 0 <= end[1] < cols else 'out of bounds'})"
    
    # Quick connectivity check using BFS
    from collections import deque
    queue = deque([start])
    visited = {start}
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        current = queue.popleft()
        
        if current == end:
            return True, "Path exists between start and end"
        
        for dr, dc in directions:
            new_row, new_col = current[0] + dr, current[1] + dc
            neighbor = (new_row, new_col)
            
            if (0 <= new_row < rows and 0 <= new_col < cols and
                grid[new_row][new_col] == 0 and neighbor not in visited):
                visited.add(neighbor)
                queue.append(neighbor)
    
    return False, f"No path exists between start {start} and end {end}"
        
def process_maze(filepath):
    """Process maze image and solve using all algorithms"""
    try:
        print(f"Processing maze from: {filepath}")
        
        # Convert image to grid
        converter = ImageToGrid()
        grid, start, end = converter.convert(filepath)
        
        print(f"Grid size: {len(grid)}x{len(grid[0]) if grid else 0}")
        print(f"Start: {start}, End: {end}")
        
        if not start or not end:
            return {'error': 'Could not detect start and end points', 'success': False}
        
        # Debug grid connectivity
        is_connected, connectivity_message = debug_grid_connectivity(grid, start, end)
        print(f"Connectivity check: {connectivity_message}")
        
        if not is_connected:
            # Try to fix connectivity by ensuring start and end are properly connected
            print("Attempting to fix connectivity...")
            
            # Make sure start and end are definitely paths
            grid[start[0]][start[1]] = 0
            grid[end[0]][end[1]] = 0
            
            # Try a simple path creation between start and end
            if not create_simple_path(grid, start, end):
                return {'error': f'Maze is not solvable: {connectivity_message}', 'success': False}
        
        # Get path folder
        path_folder = get_next_path_folder()
        print(f"Using path folder: {path_folder}")
        
        # Calculate difficulty
        difficulty = calculate_maze_difficulty(grid)
        print(f"Calculated difficulty: {difficulty}")
        
        # Load original image for overlays
        original_image = cv2.imread(filepath)
        if original_image is None:
            return {'error': 'Could not load original image', 'success': False}
        
        # Solve with all algorithms
        results = {}
        algorithm_stats = []
        
        for algo_name, algorithm in algorithms.items():
            try:
                print(f"Running {algo_name}...")
                start_time = time.time()
                
                # Solve maze
                path, visited = algorithm.solve(grid, start, end)
                
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
                
                print(f"{algo_name}: {execution_time:.2f}ms, path length: {len(path)}, visited: {len(visited)}")
                
                # Check if algorithm found a valid path
                if not path or len(path) <= 1:
                    print(f"{algo_name} failed to find a path")
                    # Create image with "No Path Found" message
                    overlay_image = original_image.copy()
                    
                    # Add "Unable to craft a path" message
                    img_height, img_width = overlay_image.shape[:2]
                    text = "Unable to craft a path"
                    font_scale = min(img_width, img_height) / 800.0  # Scale font to image size
                    font_thickness = max(1, int(font_scale * 2))
                    
                    # Get text size for centering
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    
                    # Center the text
                    x = (img_width - text_width) // 2
                    y = (img_height + text_height) // 2
                    
                    # Add semi-transparent background
                    overlay = overlay_image.copy()
                    cv2.rectangle(overlay, (x-10, y-text_height-10), (x+text_width+10, y+10), (0, 0, 0), -1)
                    cv2.addWeighted(overlay_image, 0.7, overlay, 0.3, 0, overlay_image)
                    
                    # Add text
                    cv2.putText(overlay_image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
                    
                    # Store stats for failed algorithm
                    algorithm_stats.append({
                        'algorithm': algo_name.upper(),
                        'time': f"{execution_time:.2f}ms",
                        'path_length': 0,
                        'nodes_explored': len(visited),
                        'complexity': algorithm.get_complexity(),
                        'status': 'No Path Found'
                    })
                else:
                    # Create clean overlay image (no algorithm name, no gray dots)
                    overlay_image = create_path_overlay(
                        original_image.copy(), 
                        path, 
                        visited, 
                        start, 
                        end, 
                        grid
                    )
                    
                    # Store stats for successful algorithm
                    algorithm_stats.append({
                        'algorithm': algo_name.upper(),
                        'time': f"{execution_time:.2f}ms",
                        'path_length': len(path),
                        'nodes_explored': len(visited),
                        'complexity': algorithm.get_complexity(),
                        'status': 'Path Found'
                    })
                
                # Save result image
                result_path = os.path.join(path_folder, f'{algo_name}.png')
                cv2.imwrite(result_path, overlay_image)
                
                results[algo_name] = {
                    'path': path,
                    'visited': visited,
                    'time': execution_time,
                    'image_path': result_path,
                    'success': len(path) > 1
                }
                
            except Exception as algo_error:
                print(f"Error in {algo_name}: {str(algo_error)}")
                # Create error image
                overlay_image = original_image.copy()
                error_text = "Algorithm Error"
                img_height, img_width = overlay_image.shape[:2]
                font_scale = min(img_width, img_height) / 800.0
                font_thickness = max(1, int(font_scale * 2))
                
                (text_width, text_height), _ = cv2.getTextSize(error_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                x = (img_width - text_width) // 2
                y = (img_height + text_height) // 2
                
                cv2.putText(overlay_image, error_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
                
                result_path = os.path.join(path_folder, f'{algo_name}.png')
                cv2.imwrite(result_path, overlay_image)
                
                algorithm_stats.append({
                    'algorithm': algo_name.upper(),
                    'time': '0ms',
                    'path_length': 0,
                    'nodes_explored': 0,
                    'complexity': 'N/A',
                    'status': 'Error'
                })
                continue
        
        if not results:
            return {'error': 'All algorithms failed to solve the maze', 'success': False}
        
        # Generate animation GIF
        try:
            gif_generator = GifGenerator()
            animation_path = os.path.join(path_folder, 'animation.gif')
            # Use the first successful algorithm for animation
            successful_algos = [name for name, result in results.items() if result.get('success', True)]
            if successful_algos:
                first_algo = successful_algos[0]
                gif_generator.create_pathfinding_animation(
                    original_image, 
                    results[first_algo]['path'], 
                    results[first_algo]['visited'], 
                    start, 
                    end, 
                    animation_path
                )
            else:
                animation_path = None
        except Exception as gif_error:
            print(f"Warning: Could not create animation: {str(gif_error)}")
            animation_path = None
        
        print(f"Processing complete. Successful algorithms: {len([r for r in results.values() if r.get('success', True)])}")
        
        return {
            'success': True,
            'path_folder': path_folder,
            'difficulty': difficulty,
            'stats': algorithm_stats,
            'animation_path': animation_path
        }
        
    except Exception as e:
        print(f"Error in process_maze: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'success': False}


def create_simple_path(grid, start, end):
    """
    Create a simple path between start and end if possible.
    This is a fallback when the maze seems disconnected.
    """
    try:
        rows, cols = len(grid), len(grid[0])
        
        # Simple straight-line path attempt
        current_row, current_col = start
        end_row, end_col = end
        
        # Move towards the end position
        while (current_row, current_col) != (end_row, end_col):
            # Move vertically first
            if current_row < end_row:
                current_row += 1
            elif current_row > end_row:
                current_row -= 1
            # Then move horizontally
            elif current_col < end_col:
                current_col += 1
            elif current_col > end_col:
                current_col -= 1
            
            # Ensure we don't go out of bounds
            if 0 <= current_row < rows and 0 <= current_col < cols:
                grid[current_row][current_col] = 0  # Make it a path
            else:
                break
        
        # Verify connectivity again
        is_connected, _ = debug_grid_connectivity(grid, start, end)
        return is_connected
    
    except Exception as e:
        print(f"Error creating simple path: {e}")
        return False
    
        
    except Exception as e:
        print(f"Error in process_maze: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'success': False}

def create_path_overlay(image, path, visited, start, end, grid):
    """Create clean path overlay on maze image with accurate scaling"""
    if not path:
        return image
    
    height, width = len(grid), len(grid[0])
    img_height, img_width = image.shape[:2]
    
    # Calculate scaling factors more accurately
    scale_x = img_width / width
    scale_y = img_height / height
    
    print(f"Creating overlay: Grid {height}x{width}, Image {img_height}x{img_width}")
    print(f"Scale factors: X={scale_x:.2f}, Y={scale_y:.2f}")
    print(f"Path length: {len(path)}, Start: {start}, End: {end}")
    
    # Create overlay image
    overlay = image.copy()
    
    # Draw path with thicker lines and better visibility
    if len(path) > 1:
        for i in range(len(path) - 1):
            y1, x1 = path[i]
            y2, x2 = path[i + 1]
            
            # Convert grid coordinates to image coordinates more precisely
            point1 = (
                int(x1 * scale_x + scale_x // 2), 
                int(y1 * scale_y + scale_y // 2)
            )
            point2 = (
                int(x2 * scale_x + scale_x // 2), 
                int(y2 * scale_y + scale_y // 2)
            )
            
            # Ensure points are within image bounds
            point1 = (
                max(0, min(img_width - 1, point1[0])),
                max(0, min(img_height - 1, point1[1]))
            )
            point2 = (
                max(0, min(img_width - 1, point2[0])),
                max(0, min(img_height - 1, point2[1]))
            )
            
            # Draw thick orange path line
            cv2.line(overlay, point1, point2, (0, 165, 255), 6)  # Orange path, thicker
    
    # Draw start point (green circle)
    if start:
        start_y, start_x = start
        start_center = (
            int(start_x * scale_x + scale_x // 2), 
            int(start_y * scale_y + scale_y // 2)
        )
        start_center = (
            max(0, min(img_width - 1, start_center[0])),
            max(0, min(img_height - 1, start_center[1]))
        )
        cv2.circle(overlay, start_center, 12, (0, 255, 0), -1)  # Green start
        cv2.circle(overlay, start_center, 12, (0, 0, 0), 3)     # Black border
    
    # Draw end point (red circle)
    if end:
        end_y, end_x = end
        end_center = (
            int(end_x * scale_x + scale_x // 2), 
            int(end_y * scale_y + scale_y // 2)
        )
        end_center = (
            max(0, min(img_width - 1, end_center[0])),
            max(0, min(img_height - 1, end_center[1]))
        )
        cv2.circle(overlay, end_center, 12, (0, 0, 255), -1)   # Red end
        cv2.circle(overlay, end_center, 12, (0, 0, 0), 3)      # Black border
    
    return overlay

@app.route('/download_results/<path_folder>')
def download_results(path_folder):
    """Download all results as ZIP file"""
    zip_path = f'{path_folder}.zip'
    shutil.make_archive(path_folder, 'zip', path_folder)
    return send_file(zip_path, as_attachment=True, download_name=f'{os.path.basename(path_folder)}_results.zip')

@app.route('/get_results/<path_folder>')
def get_results(path_folder):
    """Get results for a specific path folder"""
    results = []
    algorithms_order = ['bfs', 'dfs', 'astar', 'dijkstra', 'greedy', 'bidirectional_bfs']
    
    for algo in algorithms_order:
        image_path = os.path.join(path_folder, f'{algo}.png')
        if os.path.exists(image_path):
            results.append({
                'algorithm': algo.upper(),
                'image_path': image_path
            })
    
    animation_path = os.path.join(path_folder, 'animation.gif')
    
    return jsonify({
        'results': results,
        'animation_path': animation_path if os.path.exists(animation_path) else None
    })

@app.route('/debug_grid/<path:filename>')
def debug_grid(filename):
    """Debug route to visualize grid conversion"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        converter = ImageToGrid()
        grid, start, end = converter.convert(filepath)
        
        # Check connectivity
        is_connected, connectivity_message = debug_grid_connectivity(grid, start, end)
        
        debug_info = {
            'grid_size': f"{len(grid)}x{len(grid[0]) if grid else 0}",
            'start': start,
            'end': end,
            'wall_count': sum(row.count(1) for row in grid),
            'path_count': sum(row.count(0) for row in grid),
            'is_connected': is_connected,
            'connectivity_message': connectivity_message,
            'grid_sample': grid[:10] if grid else [],  # First 10 rows for inspection
            'start_area': get_area_around_point(grid, start, 3) if start else None,
            'end_area': get_area_around_point(grid, end, 3) if end else None
        }
        
        return jsonify(debug_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_area_around_point(grid, point, radius):
    """Get the area around a point for debugging"""
    if not point or not grid:
        return None
    
    rows, cols = len(grid), len(grid[0])
    row, col = point
    
    area = []
    for r in range(max(0, row - radius), min(rows, row + radius + 1)):
        area_row = []
        for c in range(max(0, col - radius), min(cols, col + radius + 1)):
            area_row.append(grid[r][c])
        area.append(area_row)
    
    return {
        'center': point,
        'area': area,
        'center_value': grid[row][col] if 0 <= row < rows and 0 <= col < cols else None
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)