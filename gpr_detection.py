# -*- coding: utf-8 -*-
"""
GPR Detection System for Webots Drone Simulation
Simulates Ground Penetrating Radar for buried object detection
Author: Generated for drone GPR integration
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import threading
import queue
import time
import os
import csv
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json

@dataclass
class ScanResult:
    """Data structure for a single GPR scan result"""
    x: float
    y: float
    z: float
    timestamp: float
    label: str
    amplitude_data: np.ndarray
    depth_profile: np.ndarray
    max_amplitude: float
    detection_confidence: float
    has_detection: bool

@dataclass
class BuriedObject:
    """Represents a buried object in the simulation"""
    x: float
    y: float
    depth: float
    size: float
    material_type: str
    reflection_strength: float

class GPRSimulator:
    """Simulates GPR responses for buried objects"""
    
    def __init__(self):
        # Define buried objects based on your world (pedestrians, debris, etc.)
        self.buried_objects = [
            # Pedestrian at (9.01436, 0, 1.76388) - simulate as buried human remains
            BuriedObject(x=9.0, y=0.0, depth=0.5, size=1.8, 
                        material_type="organic", reflection_strength=0.7),
            
            # Pedestrian at (10, 0, -2) - deeper buried object
            BuriedObject(x=10.0, y=0.0, depth=2.0, size=1.8, 
                        material_type="organic", reflection_strength=0.6),
            
            # Debris objects
            BuriedObject(x=2.34, y=9.06, depth=0.3, size=0.5, 
                        material_type="metal", reflection_strength=0.9),
            BuriedObject(x=18.1, y=17.15, depth=0.4, size=0.6, 
                        material_type="metal", reflection_strength=0.8),
            BuriedObject(x=17.84, y=9.68, depth=0.2, size=0.4, 
                        material_type="concrete", reflection_strength=0.5),
            
            # Additional simulated buried objects
            BuriedObject(x=5.0, y=5.0, depth=1.0, size=1.0, 
                        material_type="metal", reflection_strength=0.85),
            BuriedObject(x=15.0, y=-5.0, depth=0.8, size=0.8, 
                        material_type="plastic", reflection_strength=0.3),
        ]
        
        # GPR parameters
        self.frequency = 400e6  # 400 MHz
        self.depth_range = 3.0  # meters
        self.depth_resolution = 0.02  # 2cm resolution
        self.detection_radius = 2.0  # Detection radius in meters
        
        # Generate depth axis
        self.depths = np.arange(0, self.depth_range, self.depth_resolution)
        self.n_samples = len(self.depths)
        
    def simulate_gpr_scan(self, x: float, y: float, z: float) -> Tuple[np.ndarray, float, bool]:
        """
        Simulate a GPR scan at given position
        Returns: (amplitude_data, max_amplitude, has_detection)
        """
        # Initialize amplitude data with noise
        amplitude_data = np.random.normal(0, 0.01, self.n_samples)
        max_amplitude = 0.0
        has_detection = False
        
        # Add ground reflection
        ground_idx = int(abs(z) / self.depth_resolution)
        if ground_idx < self.n_samples:
            amplitude_data[ground_idx] += 0.3
        
        # Check for buried objects
        for obj in self.buried_objects:
            distance = np.sqrt((x - obj.x)**2 + (y - obj.y)**2)
            
            if distance <= self.detection_radius:
                # Calculate signal strength based on distance
                signal_strength = obj.reflection_strength * np.exp(-distance / self.detection_radius)
                
                # Add object reflection at appropriate depth
                obj_depth_idx = int(obj.depth / self.depth_resolution)
                if obj_depth_idx < self.n_samples:
                    # Add main reflection
                    amplitude_data[obj_depth_idx] += signal_strength
                    
                    # Add secondary reflections (ringing effect)
                    for i in range(1, 4):
                        secondary_idx = obj_depth_idx + i * 5
                        if secondary_idx < self.n_samples:
                            amplitude_data[secondary_idx] += signal_strength * 0.3 * (0.7**i)
                    
                    # Update detection status
                    if signal_strength > 0.15:  # Detection threshold
                        has_detection = True
                        max_amplitude = max(max_amplitude, signal_strength)
        
        # Apply realistic GPR processing (envelope detection)
        amplitude_data = np.abs(amplitude_data)
        
        return amplitude_data, max_amplitude, has_detection

class GPRDetector:
    """Main GPR detection class with threading support"""
    
    def __init__(self, gpr_template_path: str = None, gpr_output_dir: str = None, threshold: float = 0.05):
        self.threshold = threshold
        self.output_dir = gpr_output_dir or "gpr_outputs"
        self.template_path = gpr_template_path
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize GPR simulator
        self.simulator = GPRSimulator()
        
        # Threading components
        self.scan_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.processing_thread = None
        self.is_running = False
        
        # Storage for results
        self.scan_results: List[ScanResult] = []
        self.scan_lock = threading.Lock()
        
        print(f"[GPR] Initialized GPR detector with output dir: {self.output_dir}")
        print(f"[GPR] Buried objects in simulation: {len(self.simulator.buried_objects)}")
        
    def start(self):
        """Start the processing thread"""
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            print("[GPR] Processing thread started")
    
    def stop(self):
        """Stop the processing thread"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        print("[GPR] Processing thread stopped")
    
    def queue_scan(self, x: float, y: float, z: float, label: str = "scan"):
        """Queue a GPR scan for processing"""
        if not self.is_running:
            self.start()
        
        scan_data = {
            'x': x, 'y': y, 'z': z, 'label': label,
            'timestamp': time.time()
        }
        self.scan_queue.put(scan_data)
        return True
    
    def _processing_loop(self):
        """Main processing loop running in separate thread"""
        while self.is_running:
            try:
                # Get scan request with timeout
                scan_data = self.scan_queue.get(timeout=0.1)
                
                # Perform GPR simulation
                amplitude_data, max_amplitude, has_detection = self.simulator.simulate_gpr_scan(
                    scan_data['x'], scan_data['y'], scan_data['z']
                )
                
                # Create depth profile
                depth_profile = self.simulator.depths.copy()
                
                # Calculate detection confidence
                detection_confidence = min(max_amplitude / 0.5, 1.0) if has_detection else 0.0
                
                # Create scan result
                result = ScanResult(
                    x=scan_data['x'],
                    y=scan_data['y'],
                    z=scan_data['z'],
                    timestamp=scan_data['timestamp'],
                    label=scan_data['label'],
                    amplitude_data=amplitude_data,
                    depth_profile=depth_profile,
                    max_amplitude=max_amplitude,
                    detection_confidence=detection_confidence,
                    has_detection=has_detection
                )
                
                # Store result
                with self.scan_lock:
                    self.scan_results.append(result)
                
                # Put result in queue for immediate access
                self.result_queue.put(result)
                
                print(f"[GPR] Processed scan at ({scan_data['x']:.2f}, {scan_data['y']:.2f}) - "
                      f"Detection: {has_detection}, Confidence: {detection_confidence:.3f}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[GPR] Error in processing loop: {e}")
    
    def get_latest_result(self) -> Optional[ScanResult]:
        """Get the latest scan result if available"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_all_results(self) -> List[ScanResult]:
        """Get all scan results"""
        with self.scan_lock:
            return self.scan_results.copy()
    
    def get_detection_summary(self) -> Dict:
        """Get summary of all detections"""
        with self.scan_lock:
            total_scans = len(self.scan_results)
            detections = sum(1 for r in self.scan_results if r.has_detection)
            max_amplitude = max([r.max_amplitude for r in self.scan_results], default=0.0)
            
            return {
                'total_scans': total_scans,
                'detections': detections,
                'detection_rate': detections / total_scans if total_scans > 0 else 0.0,
                'max_amplitude': max_amplitude
            }

class WebotsGPRInterface:
    """Interface between Webots controller and GPR detector"""
    
    def __init__(self, gpr_detector: GPRDetector, robot_controller):
        self.gpr_detector = gpr_detector
        self.robot_controller = robot_controller
        self.scan_log = []
        
        # Start GPR detector
        self.gpr_detector.start()
        
    def scan_position(self, x: float, y: float, z: float, label: str = "scan") -> bool:
        """Perform GPR scan at specified position"""
        try:
            success = self.gpr_detector.queue_scan(x, y, z, label)
            if success:
                self.scan_log.append({
                    'x': x, 'y': y, 'z': z, 'label': label,
                    'timestamp': time.time()
                })
            return success
        except Exception as e:
            print(f"[GPR] Error scanning position: {e}")
            return False
    
    def get_latest_detection(self) -> Optional[ScanResult]:
        """Get latest scan result"""
        return self.gpr_detector.get_latest_result()
    
    def get_detection_summary(self) -> Dict:
        """Get detection summary"""
        return self.gpr_detector.get_detection_summary()
    
    def plot_detection_map(self, save_path: str = None):
        """Generate detection map visualization"""
        results = self.gpr_detector.get_all_results()
        if not results:
            print("[GPR] No scan data available for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot scan positions
        x_coords = [r.x for r in results]
        y_coords = [r.y for r in results]
        detections = [r.has_detection for r in results]
        confidences = [r.detection_confidence for r in results]
        
        # Create scatter plot with color coding
        colors = ['red' if d else 'blue' for d in detections]
        sizes = [50 + 200 * c for c in confidences]  # Size based on confidence
        
        scatter = ax.scatter(x_coords, y_coords, c=colors, s=sizes, alpha=0.6)
        
        # Plot buried objects for reference
        for obj in self.gpr_detector.simulator.buried_objects:
            circle = patches.Circle((obj.x, obj.y), obj.size/2, 
                                  fill=False, edgecolor='green', linewidth=2, linestyle='--')
            ax.add_patch(circle)
            ax.text(obj.x, obj.y + obj.size/2 + 0.3, f'{obj.material_type}\n({obj.depth:.1f}m)', 
                   ha='center', va='bottom', fontsize=8, color='green')
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('GPR Detection Map\nRed=Detection, Blue=No Detection, Green=Actual Objects')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=10, label='Detection'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=10, label='No Detection'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
                      markeredgecolor='green', markersize=10, label='Buried Object')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.gpr_detector.output_dir, 'detection_map.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
        print(f"[GPR] Detection map saved")
    
    def plot_full_flight_log(self, save_path: str = None):
        """Plot the full flight path with scan locations"""
        if not self.scan_log:
            print("[GPR] No flight log data available")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract coordinates
        x_coords = [entry['x'] for entry in self.scan_log]
        y_coords = [entry['y'] for entry in self.scan_log]
        
        # Plot flight path
        ax.plot(x_coords, y_coords, 'b-', alpha=0.5, linewidth=2, label='Flight Path')
        ax.scatter(x_coords, y_coords, c='blue', s=30, alpha=0.7, label='Scan Points')
        
        # Add start and end markers
        if len(x_coords) > 0:
            ax.scatter(x_coords[0], y_coords[0], c='green', s=100, marker='^', 
                      label='Start', zorder=5)
            ax.scatter(x_coords[-1], y_coords[-1], c='red', s=100, marker='v', 
                      label='End', zorder=5)
        
        # Plot buried objects
        for obj in self.gpr_detector.simulator.buried_objects:
            circle = patches.Circle((obj.x, obj.y), obj.size/2, 
                                  fill=False, edgecolor='orange', linewidth=2)
            ax.add_patch(circle)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('GPR Mission Flight Log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.gpr_detector.output_dir, 'flight_log.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
        print(f"[GPR] Flight log plot saved")
    
    def write_log_to_csv(self, filepath: str = None):
        """Write scan results to CSV file"""
        results = self.gpr_detector.get_all_results()
        if not results:
            print("[GPR] No data to write to CSV")
            return
        
        if filepath is None:
            filepath = os.path.join(self.gpr_detector.output_dir, 'gpr_scan_log.csv')
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'x', 'y', 'z', 'label', 'has_detection', 
                         'detection_confidence', 'max_amplitude']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow({
                    'timestamp': datetime.fromtimestamp(result.timestamp).isoformat(),
                    'x': result.x,
                    'y': result.y,
                    'z': result.z,
                    'label': result.label,
                    'has_detection': result.has_detection,
                    'detection_confidence': result.detection_confidence,
                    'max_amplitude': result.max_amplitude
                })
        
        print(f"[GPR] CSV log written to {filepath}")
    
    def shutdown(self):
        """Shutdown the GPR system"""
        print("[GPR] Shutting down GPR interface...")
        self.gpr_detector.stop()
        
        # Generate final summary
        summary = self.get_detection_summary()
        print(f"[GPR] Final Summary: {summary}")

if __name__ == "__main__":
    # Test the GPR system
    print("Testing GPR Detection System...")
    
    # Initialize detector
    detector = GPRDetector()
    
    # Create mock robot controller
    class MockController:
        def get_position(self):
            return (0, 0, 1)
    
    # Initialize interface
    gpr = WebotsGPRInterface(detector, MockController())
    
    # Perform some test scans
    test_positions = [
        (0, 0, 1), (2, 2, 1), (5, 0, 1), (9, 0, 1),  # Near buried objects
        (10, 0, 1), (15, 15, 1), (20, 20, 1)
    ]
    
    for i, (x, y, z) in enumerate(test_positions):
        gpr.scan_position(x, y, z, f"test_scan_{i}")
        time.sleep(0.1)  # Small delay
    
    # Wait for processing
    time.sleep(2)
    
    # Generate reports
    summary = gpr.get_detection_summary()
    print(f"Test Summary: {summary}")
    
    gpr.plot_detection_map()
    gpr.write_log_to_csv()
    
    # Shutdown
    gpr.shutdown()