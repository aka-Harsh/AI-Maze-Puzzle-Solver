
class MazeSolverApp {
    constructor() {
        this.currentResults = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupDragAndDrop();
        this.setupSmoothScrolling();
    }

    setupEventListeners() {
        // File upload
        const fileInput = document.getElementById('mazeFile');
        const uploadArea = document.getElementById('uploadArea');
        const processMaze = document.getElementById('processMaze');
        const generateMaze = document.getElementById('generateMaze');
        const downloadResults = document.getElementById('downloadResults');

        // Upload area click
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        // File input change
        fileInput.addEventListener('change', (e) => {
            this.handleFileSelect(e.target.files[0]);
        });

        // Process maze button
        processMaze.addEventListener('click', () => {
            this.processMaze();
        });

        // Generate maze button
        generateMaze.addEventListener('click', () => {
            this.generateMaze();
        });

        // Download results
        downloadResults.addEventListener('click', () => {
            this.downloadResults();
        });

        // Maze dimension inputs (ensure odd numbers)
        document.getElementById('mazeWidth').addEventListener('change', (e) => {
            this.ensureOddNumber(e.target);
        });

        document.getElementById('mazeHeight').addEventListener('change', (e) => {
            this.ensureOddNumber(e.target);
        });
    }

    setupDragAndDrop() {
        const uploadArea = document.getElementById('uploadArea');

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.remove('dragover');
            }, false);
        });

        uploadArea.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        }, false);
    }

    setupSmoothScrolling() {
        // Smooth scrolling for navigation links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ensureOddNumber(input) {
        let value = parseInt(input.value);
        if (value % 2 === 0) {
            input.value = value + 1;
        }
    }

    handleFileSelect(file) {
        if (!file) return;

        // Validate file type
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
        if (!validTypes.includes(file.type)) {
            this.showNotification('Please select a PNG or JPEG image file.', 'error');
            return;
        }

        // Validate file size (16MB max)
        const maxSize = 16 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showNotification('File size must be less than 16MB.', 'error');
            return;
        }

        // Update UI
        const uploadContent = document.querySelector('.upload-content');
        uploadContent.innerHTML = `
            <i class="bi bi-check-circle display-1 text-success mb-3"></i>
            <h4>File Selected: ${file.name}</h4>
            <p class="text-muted">Size: ${this.formatFileSize(file.size)}</p>
            <p class="small text-muted">Ready to process!</p>
        `;

        // Enable process button
        document.getElementById('processMaze').disabled = false;
        this.selectedFile = file;

        this.showNotification('File uploaded successfully!', 'success');
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async processMaze() {
        console.log('processMaze called');
        
        if (!this.selectedFile) {
            this.showNotification('Please select a maze file first.', 'error');
            return;
        }

        try {
            console.log('Starting maze processing...');
            this.showLoading(true);

            const formData = new FormData();
            formData.append('maze_file', this.selectedFile);

            console.log('Sending request to /upload');
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            console.log('Response received:', response.status);
            const result = await response.json();
            console.log('Result:', result);

            if (result.success) {
                this.currentResults = result;
                this.displayResults(result);
                this.showNotification('Maze solved successfully!', 'success');
            } else {
                this.showNotification(result.error || 'Failed to process maze.', 'error');
            }
        } catch (error) {
            console.error('Error processing maze:', error);
            this.showNotification('An error occurred while processing the maze.', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    async generateMaze() {
        console.log('generateMaze called');
        
        const width = parseInt(document.getElementById('mazeWidth').value);
        const height = parseInt(document.getElementById('mazeHeight').value);

        try {
            console.log(`Generating maze: ${width}x${height}`);
            this.showLoading(true);

            console.log('Sending request to /generate_maze');
            const response = await fetch('/generate_maze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ width, height })
            });

            console.log('Response received:', response.status);
            const result = await response.json();
            console.log('Result:', result);

            if (result.success) {
                this.currentResults = result;
                this.displayResults(result);
                
                // Update upload area to show generated maze
                const uploadContent = document.querySelector('.upload-content');
                if (uploadContent) {
                    uploadContent.innerHTML = `
                        <i class="bi bi-check-circle display-1 text-success mb-3"></i>
                        <h4>Maze Generated (${width}x${height})</h4>
                        <p class="text-muted">Maze created and solved successfully!</p>
                    `;
                }

                this.showNotification('Maze generated and solved successfully!', 'success');
            } else {
                this.showNotification(result.error || 'Failed to generate maze.', 'error');
            }
        } catch (error) {
            console.error('Error generating maze:', error);
            this.showNotification('An error occurred while generating the maze.', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    displayResults(result) {
        try {
            // Show results sections
            const resultsSection = document.getElementById('results-section');
            const statsSection = document.getElementById('stats-section');
            
            if (resultsSection) resultsSection.classList.remove('d-none');
            if (statsSection) statsSection.classList.remove('d-none');

            // Update difficulty
            const difficultyLevel = document.getElementById('difficultyLevel');
            const difficultyBadge = document.getElementById('difficultyBadge');
            
            if (difficultyLevel && difficultyBadge && result.difficulty) {
                difficultyLevel.textContent = result.difficulty;
                difficultyBadge.className = `badge fs-6 difficulty-${result.difficulty.toLowerCase()}`;
            }

            // Generate algorithm cards
            if (result.path_folder) {
                this.generateAlgorithmCards(result.path_folder);
            }

            // Populate statistics table
            if (result.stats) {
                this.populateStatsTable(result.stats);
            }

            // Set animation
            if (result.animation_path) {
                const animationGif = document.getElementById('animationGif');
                if (animationGif) {
                    animationGif.src = result.animation_path;
                }
            }

            // Scroll to results
            setTimeout(() => {
                const resultsSection = document.getElementById('results-section');
                if (resultsSection) {
                    resultsSection.scrollIntoView({
                        behavior: 'smooth'
                    });
                }
            }, 500);

        } catch (error) {
            console.error('Error displaying results:', error);
            this.showNotification('Error displaying results. Please try again.', 'error');
        }
    }

    generateAlgorithmCards(pathFolder) {
        const algorithms = [
            { name: 'BFS', fullName: 'Breadth-First Search', icon: 'bi-arrow-right-circle', color: 'primary' },
            { name: 'DFS', fullName: 'Depth-First Search', icon: 'bi-arrow-down-circle', color: 'success' },
            { name: 'A*', fullName: 'A* Search', icon: 'bi-star', color: 'warning' },
            { name: 'Dijkstra', fullName: 'Dijkstra\'s Algorithm', icon: 'bi-diagram-3', color: 'info' },
            { name: 'Greedy', fullName: 'Greedy Best-First', icon: 'bi-lightning', color: 'danger' },
            { name: 'Bidirectional BFS', fullName: 'Bidirectional BFS', icon: 'bi-arrow-left-right', color: 'secondary' }
        ];

        const grid = document.getElementById('algorithmGrid');
        grid.innerHTML = '';

        algorithms.forEach((algo, index) => {
            const imagePath = `${pathFolder}/${algo.name.toLowerCase().replace('*', 'star').replace(' ', '_')}.png`;
            
            const card = document.createElement('div');
            card.className = 'col-lg-4 col-md-6';
            card.innerHTML = `
                <div class="card algorithm-card shadow-lg animate-slide-up" style="animation-delay: ${index * 0.1}s">
                    <div class="algorithm-badge">
                        <span class="badge bg-${algo.color}">
                            <i class="${algo.icon}"></i> ${algo.name}
                        </span>
                    </div>
                    <img src="${imagePath}" class="card-img-top" alt="${algo.fullName} Result">
                    <div class="card-body">
                        <h5 class="card-title">
                            <i class="${algo.icon}"></i> ${algo.fullName}
                        </h5>
                        <p class="card-text text-muted">
                            ${this.getAlgorithmDescription(algo.name)}
                        </p>
                        <div class="d-flex justify-content-between align-items-center">
                            <button class="btn btn-outline-${algo.color} btn-sm" onclick="app.viewFullImage('${imagePath}')">
                                <i class="bi bi-eye"></i> View Full
                            </button>
                            <small class="text-muted">
                                <i class="bi bi-cpu"></i> Algorithm
                            </small>
                        </div>
                    </div>
                </div>
            `;

            grid.appendChild(card);
        });
    }

    getAlgorithmDescription(algoName) {
        const descriptions = {
            'BFS': 'Explores all neighbors before moving deeper. Guarantees shortest path.',
            'DFS': 'Explores as far as possible before backtracking. May not find shortest path.',
            'A*': 'Uses heuristics to guide search. Optimal with admissible heuristic.',
            'Dijkstra': 'Finds shortest path with weighted edges. Optimal for positive weights.',
            'Greedy': 'Uses heuristics only. Fast but may not find optimal solution.',
            'Bidirectional BFS': 'Searches from both start and end simultaneously.'
        };
        return descriptions[algoName] || 'Advanced pathfinding algorithm.';
    }

    populateStatsTable(stats) {
        const tbody = document.getElementById('statsTableBody');
        if (!tbody) return;
        
        tbody.innerHTML = '';

        stats.forEach((stat, index) => {
            const row = document.createElement('tr');
            row.className = 'animate-slide-in';
            row.style.animationDelay = `${index * 0.1}s`;
            
            // Color code based on status
            let statusColor = 'success';
            let pathDisplay = stat.path_length;
            
            if (stat.status === 'No Path Found') {
                statusColor = 'warning';
                pathDisplay = 'No Path';
            } else if (stat.status === 'Error') {
                statusColor = 'danger';
                pathDisplay = 'Error';
            }
            
            row.innerHTML = `
                <td>
                    <strong>${stat.algorithm}</strong>
                    ${stat.status && stat.status !== 'Path Found' ? 
                        `<br><small class="text-${statusColor}">${stat.status}</small>` : ''}
                </td>
                <td>
                    <span class="badge bg-light text-dark">${stat.time}</span>
                </td>
                <td>
                    <span class="badge bg-${statusColor}">${pathDisplay}</span>
                </td>
                <td>
                    <span class="badge bg-info">${stat.nodes_explored}</span>
                </td>
                <td>
                    <code class="text-muted">${stat.complexity}</code>
                </td>
            `;

            tbody.appendChild(row);
        });
    }

    generateAlgorithmCards(pathFolder) {
        const algorithms = [
            { name: 'BFS', fullName: 'Breadth-First Search', icon: 'bi-arrow-right-circle', color: 'primary' },
            { name: 'DFS', fullName: 'Depth-First Search', icon: 'bi-arrow-down-circle', color: 'success' },
            { name: 'A*', fullName: 'A* Search', icon: 'bi-star', color: 'warning' },
            { name: 'Dijkstra', fullName: 'Dijkstra\'s Algorithm', icon: 'bi-diagram-3', color: 'info' },
            { name: 'Greedy', fullName: 'Greedy Best-First', icon: 'bi-lightning', color: 'danger' },
            { name: 'Bidirectional BFS', fullName: 'Bidirectional BFS', icon: 'bi-arrow-left-right', color: 'secondary' }
        ];

        const grid = document.getElementById('algorithmGrid');
        if (!grid) return;
        
        grid.innerHTML = '';

        algorithms.forEach((algo, index) => {
            const imagePath = `${pathFolder}/${algo.name.toLowerCase().replace('*', 'star').replace(' ', '_')}.png`;
            
            const card = document.createElement('div');
            card.className = 'col-lg-4 col-md-6';
            card.innerHTML = `
                <div class="card algorithm-card shadow-lg animate-slide-up" style="animation-delay: ${index * 0.1}s">
                    <div class="algorithm-badge">
                        <span class="badge bg-${algo.color}">
                            <i class="${algo.icon}"></i> ${algo.name}
                        </span>
                    </div>
                    <img src="${imagePath}" class="card-img-top" alt="${algo.fullName} Result" 
                         onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                    <div class="alert alert-warning m-3" style="display: none;">
                        <i class="bi bi-exclamation-triangle"></i> Result not available
                    </div>
                    <div class="card-body">
                        <h5 class="card-title">
                            <i class="${algo.icon}"></i> ${algo.fullName}
                        </h5>
                        <p class="card-text text-muted">
                            ${this.getAlgorithmDescription(algo.name)}
                        </p>
                        <div class="d-flex justify-content-between align-items-center">
                            <button class="btn btn-outline-${algo.color} btn-sm" onclick="app.viewFullImage('${imagePath}')">
                                <i class="bi bi-eye"></i> View Full
                            </button>
                            <small class="text-muted">
                                <i class="bi bi-cpu"></i> Algorithm
                            </small>
                        </div>
                    </div>
                </div>
            `;

            grid.appendChild(card);
        });
    }

    viewFullImage(imagePath) {
        // Create modal for full image view
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog modal-lg modal-dialog-centered">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Algorithm Result</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body text-center">
                        <img src="${imagePath}" class="img-fluid" alt="Algorithm Result">
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();

        // Remove modal from DOM when hidden
        modal.addEventListener('hidden.bs.modal', () => {
            modal.remove();
        });
    }

    downloadResults() {
        if (!this.currentResults || !this.currentResults.path_folder) {
            this.showNotification('No results to download.', 'error');
            return;
        }

        const pathFolder = this.currentResults.path_folder.split('/').pop();
        window.open(`/download_results/${pathFolder}`, '_blank');
        this.showNotification('Downloading results...', 'info');
    }

    showLoading(show) {
        const processMaze = document.getElementById('processMaze');
        
        if (!processMaze) {
            console.error('Process maze button not found');
            return;
        }

        if (show) {
            processMaze.disabled = true;
            processMaze.innerHTML = `
                <span class="spinner-border spinner-border-sm me-2"></span>
                <i class="bi bi-cpu"></i> Processing...
            `;

            // Show loading overlay
            const existingOverlay = document.getElementById('loadingOverlay');
            if (!existingOverlay) {
                const overlay = document.createElement('div');
                overlay.className = 'loading-overlay';
                overlay.id = 'loadingOverlay';
                overlay.innerHTML = `
                    <div class="text-center">
                        <div class="loading-spinner mb-3"></div>
                        <h4 class="text-primary">Solving Maze...</h4>
                        <p class="text-muted">Running 6 algorithms simultaneously</p>
                    </div>
                `;
                document.body.appendChild(overlay);
            }
        } else {
            processMaze.disabled = false;
            processMaze.innerHTML = `
                <i class="bi bi-cpu"></i> Solve Maze
            `;

            // Remove loading overlay
            const overlay = document.getElementById('loadingOverlay');
            if (overlay) {
                overlay.remove();
            }
        }
    }

    showNotification(message, type = 'info') {
        // Create notification toast
        const toastContainer = document.getElementById('toastContainer') || this.createToastContainer();
        
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type === 'error' ? 'danger' : type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    <i class="bi bi-${this.getNotificationIcon(type)} me-2"></i>
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;

        toastContainer.appendChild(toast);
        
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();

        // Remove toast from DOM when hidden
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    }

    createToastContainer() {
        const container = document.createElement('div');
        container.id = 'toastContainer';
        container.className = 'toast-container position-fixed top-0 end-0 p-3';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
        return container;
    }

    getNotificationIcon(type) {
        const icons = {
            'success': 'check-circle',
            'error': 'exclamation-triangle',
            'warning': 'exclamation-triangle',
            'info': 'info-circle'
        };
        return icons[type] || 'info-circle';
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new MazeSolverApp();
});

// Add some utility functions for animations
function addScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate__animated', 'animate__fadeInUp');
            }
        });
    });

    document.querySelectorAll('.card, .table, .badge').forEach(el => {
        observer.observe(el);
    });
}

// Initialize scroll animations
document.addEventListener('DOMContentLoaded', addScrollAnimations);

