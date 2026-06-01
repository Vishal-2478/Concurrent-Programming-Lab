# 🚀 Concurrent Programming with CUDA

This repository contains CUDA-based parallel computing practicals implemented using **Python**, **Numba CUDA**, and **Google Colab (Tesla T4 GPU)**.

The objective of these practicals is to demonstrate how GPU parallelism can accelerate vector processing, image processing, and graph analytics workloads.

---

## 📚 Practicals Included

### 1️⃣ Count Even Numbers Using CUDA

Count the total number of even numbers in a large vector using GPU parallel processing.

#### Dataset

* DS1 Vector Operations Dataset
* Randomly generated integer vector
* Size: 10,000,000 elements

#### Concepts Used

* CUDA Kernel
* Threads
* Blocks
* Grids
* Atomic Operations
* Parallel Processing

#### Workflow

```text
Input Vector
      ↓
Copy Data To GPU
      ↓
Launch CUDA Kernel
      ↓
Each Thread Processes One Element
      ↓
Check Number % 2 == 0
      ↓
Atomic Counter Update
      ↓
Copy Result To CPU
      ↓
Display Total Even Count
```

---

### 2️⃣ RGB to Grayscale Conversion Using CUDA

Convert RGB images into grayscale using GPU parallel processing.

#### Dataset

**CIFAR-10 Dataset**

* 60,000 RGB Images
* Resolution: 32 × 32
* 10 Object Classes

#### Grayscale Formula

```text
Gray = 0.299R + 0.587G + 0.114B
```

#### Concepts Used

* CUDA Kernel
* 2D Grid
* Pixel-Level Parallelism
* GPU Memory Management

#### Workflow

```text
Load CIFAR-10 Image
          ↓
Transfer Image To GPU
          ↓
Launch CUDA Kernel
          ↓
Each Thread Processes One Pixel
          ↓
Extract RGB Values
          ↓
Apply Grayscale Formula
          ↓
Store Gray Pixel
          ↓
Copy Image Back To CPU
          ↓
Display Result
```

---

### 3️⃣ PageRank Implementation Using CUDA

Compute PageRank scores for nodes in a graph using GPU-accelerated parallel processing.

#### Dataset

**SNAP CollegeMsg Dataset**

Dataset Source:

https://snap.stanford.edu/data/CollegeMsg.html

Dataset Statistics:

* Nodes: 1,899
* Edges: 59,835

Columns:

```text
Source Node
Destination Node
Timestamp
```

Timestamp values are ignored during PageRank computation.

#### PageRank Formula

```text
PR(v) = (1-d)/N + d × Σ(PR(u)/OutDegree(u))
```

Where:

* PR(v) = PageRank of node v
* d = Damping Factor (0.85)
* N = Total Number of Nodes

#### Concepts Used

* Graph Processing
* CUDA Kernel
* Atomic Operations
* Parallel Edge Processing
* GPU Memory Transfer

#### Workflow

```text
Load Graph Dataset
         ↓
Create Edge List
         ↓
Compute Out Degree
         ↓
Initialize PageRank Values
         ↓
Transfer Data To GPU
         ↓
Launch CUDA Kernel
         ↓
Process Edges In Parallel
         ↓
Update Rank Values
         ↓
Repeat Iterations
         ↓
Copy Result To CPU
         ↓
Display Top Ranked Nodes
```

---

## 🖥️ Technology Stack

* Python
* CUDA
* Numba
* NumPy
* Matplotlib
* Google Colab
* NVIDIA Tesla T4 GPU

---

## ⚡ CUDA Concepts Covered

### CUDA

Compute Unified Device Architecture developed by NVIDIA for GPU programming.

### Thread

Smallest execution unit in CUDA.

### Block

Group of threads.

### Grid

Collection of blocks.

### Warp

Group of 32 CUDA threads executed simultaneously.

### Atomic Operation

Ensures safe updates to shared memory and prevents race conditions.

### Synchronization

Coordinates execution among threads and GPU operations.

### Occupancy

Measure of GPU resource utilization.

### Latency Hiding

Technique where GPU executes other warps while some warps wait for memory operations.

### Double Buffering

Technique that overlaps computation and data transfer using two buffers.

---

## 📊 Results

| Practical          | Dataset         | Output           |
| ------------------ | --------------- | ---------------- |
| Count Even Numbers | Vector Dataset  | Total Even Count |
| RGB to Grayscale   | CIFAR-10        | Grayscale Image  |
| PageRank           | SNAP CollegeMsg | Top Ranked Nodes |

---

## 🎯 Learning Outcomes

* Understanding GPU Parallel Computing
* CUDA Programming with Numba
* Vector Processing on GPU
* Image Processing using CUDA
* Graph Analytics using CUDA
* Atomic Operations and Synchronization
* Performance Optimization Techniques

---

## 👨‍💻 Author

**Vishal Mali**

Concurrent Programming Practical Assignment

Roll No: **25201309**
