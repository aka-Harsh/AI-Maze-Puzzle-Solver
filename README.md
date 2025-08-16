# 🧠 Maze Solver AI - Advanced Pathfinding Web Application

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple.svg)
![Status](https://img.shields.io/badge/status-active-green.svg)

A project gives a web application that transforms maze images into interactive pathfinding challenges. Powered by 6 advanced algorithms including A*, BFS, DFS, Dijkstra, Greedy Best-First, and Bidirectional BFS, this platform provides real-time algorithm comparison, animated visualizations, and comprehensive performance analytics. Built with Flask, OpenCV, and modern web technologies for educational excellence and algorithmic research.

---

## 🎥 Demo Video

> *Coming Soon*

---

## ✨ Features

### 🧠 **Advanced Algorithm Suite**
- **6 Pathfinding Algorithms**: BFS, DFS, A*, Dijkstra, Greedy Best-First, and Bidirectional BFS
- **Real-time Performance Comparison**: Execution time, path length, nodes explored, and complexity analysis
- **Optimal vs Heuristic**: Compare guaranteed optimal algorithms with faster heuristic approaches
- **Algorithm Visualization**: Step-by-step animated pathfinding process with GIF generation
- **Comprehensive Statistics**: Detailed performance metrics and algorithm characteristics

### 🎨 **Intelligent Image Processing**
- **Smart Grid Conversion**: Automatic maze image to binary grid conversion using OpenCV
- **Adaptive Cell Detection**: Dynamic cell size detection based on image characteristics
- **Start/End Point Recognition**: Color-based detection of green start and red end markers
- **Noise Reduction**: Advanced morphological operations for clean grid extraction
- **Multiple Format Support**: PNG, JPG, JPEG with 16MB maximum file size

### 🚀 **Interactive Web Interface**
- **Modern Responsive Design**: Bootstrap 5 with glassmorphism effects and smooth animations
- **Drag & Drop Upload**: Intuitive file handling with real-time feedback
- **Maze Generation**: Procedural maze creation using recursive backtracking
- **Results Dashboard**: 2x3 algorithm card grid with detailed visualizations
- **Download System**: ZIP packages containing all algorithm results and animations

### 🔥 **Visual Excellence**
- **Clean Path Visualization**: Thick, visible paths without visual clutter
- **Color-Coded Results**: Green start, red end, orange paths with clear markers
- **Animated Pathfinding**: Step-by-step GIF animations showing algorithm exploration
- **Performance Graphics**: Real-time statistics tables with color-coded status indicators
- **Professional UI**: Gradient backgrounds, shadow effects, and micro-animations

### 📊 **Educational Analytics**
- **Difficulty Assessment**: Automatic maze difficulty rating (Easy/Medium/Hard)
- **Algorithm Comparison**: Side-by-side performance metrics and complexity analysis
- **Learning Dashboard**: Algorithm descriptions, use cases, and limitations
- **Debug Tools**: Grid conversion inspector and connectivity analysis
- **Export Capabilities**: Download all results for further analysis

### ⚡ **High-Performance Engine**
- **Concurrent Processing**: All 6 algorithms run simultaneously for speed
- **Optimized Algorithms**: Efficient implementations with proper tie-breaking
- **Memory Management**: Smart resource utilization and cleanup
- **Error Handling**: Robust fallbacks and connectivity fixing
- **Debug Infrastructure**: Comprehensive logging and troubleshooting tools

---

## 📋 Prerequisites

### **System Requirements**
- **Python 3.8+** (3.10+ recommended for optimal performance)
- **4GB RAM** minimum (8GB recommended for large mazes)
- **Modern Web Browser** (Chrome, Firefox, Safari, Edge)
- **2GB Storage** for dependencies and result files
- **Network Connection** for CDN resources (Bootstrap, icons)

### **Optional but Recommended**
- **SSD Storage** for faster image processing
- **Multi-core Processor** (4+ cores) for concurrent algorithm execution
- **High-resolution Display** for detailed maze visualization
- **External Maze Generator** access for creating custom test cases

---

## 🚀 Quick Setup

### **1. Repository Setup**
```bash
# Clone the repository
git clone https://github.com/yourusername/maze-solver-ai.git
cd maze-solver-ai

# Verify Python version
python --version  # Should be 3.8+
```

### **2. Project Structure Creation**
```bash
# Create complete directory structure
mkdir -p static/{css,js,images,paths,uploads}
mkdir -p templates utils path_algorithms

# Create algorithm result directories
mkdir -p static/paths/{Path1,Path2,Path3}

# Verify structure
ls -la static/
```

### **3. Environment Configuration**
```bash
# Create isolated environment (recommended)
python -m venv maze_env

# Activate environment
# Windows:
maze_env\Scripts\activate
# macOS/Linux:
source maze_env/bin/activate
```

### **4. Dependency Installation**
```bash
# Install core dependencies
pip install --upgrade pip
pip install flask opencv-python numpy matplotlib pillow imageio requests

# Verify installation
python -c "import flask, cv2, numpy; print('✅ All dependencies installed')"
```

### **5. Application Launch**
```bash
# Start the Maze Solver AI
python app.py

# Access the application
# Browser: http://localhost:5000
```

---

## 🎯 Algorithm Performance Matrix

### **Algorithm Characteristics:**
| Algorithm | Time Complexity | Space | Optimal | Complete | Best Use Case |
|-----------|----------------|--------|---------|-----------|---------------|
| **BFS** | O(V + E) | O(V) | ✅ | ✅ | Shortest path in unweighted graphs |
| **DFS** | O(V + E) | O(h) | ❌ | ✅ | Memory-constrained environments |
| **A*** | O(b^d) | O(b^d) | ✅* | ✅ | Heuristic-guided optimal pathfinding |
| **Dijkstra** | O((V+E)logV) | O(V) | ✅ | ✅ | Weighted graphs and network routing |
| **Greedy** | O(b^m) | O(b^m) | ❌ | ❌ | Fast approximation when speed matters |
| **Bi-BFS** | O(b^(d/2)) | O(b^(d/2)) | ✅ | ✅ | Long paths in large search spaces |

*\*Optimal with admissible heuristic*

---

## ⚡ Performance Optimization

### **Hardware Performance Guide**
| System Configuration | Maze Size | Expected Performance | Recommended Use |
|----------------------|-----------|---------------------|-----------------|
| **Basic (4GB RAM)** | 50x50 pixels | 1-3 seconds | Educational demos |
| **Standard (8GB RAM)** | 200x200 pixels | 3-10 seconds | Classroom use |
| **High-end (16GB+)** | 500x500+ pixels | 10-30 seconds | Research applications |
| **Server Deploy** | Unlimited | <1 minute | Production environment |

### **Optimization Settings**
```bash
# High Performance Mode
FLASK_ENV=production python app.py

# Debug Mode (development)
FLASK_ENV=development FLASK_DEBUG=1 python app.py

# Memory Optimization
python app.py --max-maze-size 1000 --cache-results

# Batch Processing
python app.py --concurrent-limit 3 --timeout 60
```

### **Performance Tuning Tips**
- **Image Size**: Resize large images to 800x800 max for faster processing
- **Algorithm Selection**: Disable unused algorithms for specific use cases
- **Caching**: Enable result caching for repeated maze testing
- **Resource Limits**: Set memory and time limits for production environments

---

## 🔭 Project Outlook

<img width="1919" height="867" alt="Image" src="https://github.com/user-attachments/assets/25045ae5-7d3b-4472-a88a-4b8efd3239f4" />
<img width="1919" height="861" alt="Image" src="https://github.com/user-attachments/assets/d302f086-4531-4c1a-8b20-262912ad11ef" />
<img width="1919" height="965" alt="Image" src="https://github.com/user-attachments/assets/ba36c44f-4437-4b93-823a-3c52c9c7bb72" />
<img width="1919" height="967" alt="Image" src="https://github.com/user-attachments/assets/a8ad1f16-9a22-4d31-b5dc-0bb6c9dc60a5" />
