import sys
import os
import time
import argparse
import random
import gc
import numpy as np

# Make sure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mas.utils.animal_dataset import AnimalDataset
from mas.adapters.frame_selection_adapter import FrameSelectionAdapter
from mas.adapters.inference_adapter import InferenceAdapter
from mas.adapters.data_enhance_adapter import DataEnhanceAdapter

# Predefined paths
SELECTION_MODEL_PATH = "infra/models/frame_selector.tflite"
WEIGHT_MODEL_PATH = "infra/models/sheep_weight_predictor.tflite"

def run_benchmark(tag: str | None, num_animals: int | None, model_type: str):
    # 1. Load dataset
    dataset = AnimalDataset("data/exp1")
    all_tags = dataset.list_tags()
    
    if tag:
        if tag not in all_tags:
            print(f"[ERROR] Tag '{tag}' not found in dataset.")
            sys.exit(1)
        selected_tags = [tag]
        print(f"Benchmarking on specific animal: {tag}")
    else:
        num = num_animals if num_animals is not None else 20
        # Randomly select animals
        selected_tags = random.sample(all_tags, min(len(all_tags), num))
        print(f"Benchmarking on {len(selected_tags)} random animals: {selected_tags}")
        
    # Benchmark Selection Model
    if model_type in ["selection", "all"]:
        print("\n====================================================")
        print("--- Benchmarking Selection Model (FrameSelector) ---")
        print("====================================================")
        
        # Load model and delegate
        start_load = time.perf_counter()
        selection_adapter = FrameSelectionAdapter(suitable_window=None, model_path=SELECTION_MODEL_PATH)
        selection_adapter.load_model()
        load_time = time.perf_counter() - start_load
        print(f"Model loaded and CPU/XNNPACK delegates allocated in {load_time * 1000:.2f} ms")
        
        # Load all frames to benchmark from selected animals
        print("Loading depth images into RAM...")
        images = []
        for t in selected_tags:
            frames = dataset.load_index(t)
            for f in frames:
                img = dataset.load_depth(t, f["depth_filename"])
                images.append(img)
                
        print(f"Loaded a total of {len(images)} frames from {len(selected_tags)} animals.")
        
        # Warmup
        selection_adapter.evaluate(0.0, images[0])
        
        print("Running selection inferences...")
        latencies = []
        suitable_count = 0
        
        start_bench = time.perf_counter()
        for img in images:
            start_inf = time.perf_counter()
            is_suitable = selection_adapter.evaluate(0.0, img)
            inf_time = time.perf_counter() - start_inf
            latencies.append(inf_time)
            if is_suitable:
                suitable_count += 1
        total_bench_time = time.perf_counter() - start_bench
                
        latencies_ms = np.array(latencies) * 1000
        print("\n--- RESULTS ---")
        print(f"Total Frames Processed: {len(images)}")
        print(f"Suitable Frames Found:  {suitable_count} ({suitable_count/len(images)*100:.1f}%)")
        print(f"Total Inference Time:   {total_bench_time:.2f} seconds")
        print(f"Average Latency:        {np.mean(latencies_ms):.2f} ms")
        print(f"Min Latency:            {np.min(latencies_ms):.2f} ms")
        print(f"Max Latency:            {np.max(latencies_ms):.2f} ms")
        print(f"Std Dev:                {np.std(latencies_ms):.2f} ms")
        
        # Explicitly release memory and interpreter
        del selection_adapter
        del images
        gc.collect()
        print("Selection model and image memory released.")
        
    # Benchmark Inference/Prediction Model
    if model_type in ["prediction", "all"]:
        print("\n====================================================")
        print("--- Benchmarking Weight Prediction Model (SheepWeightPredictor) ---")
        print("====================================================")
        
        # Load model and delegate
        start_load = time.perf_counter()
        inference_adapter = InferenceAdapter(WEIGHT_MODEL_PATH)
        inference_adapter.load_model()
        load_time = time.perf_counter() - start_load
        print(f"Model loaded and CPU/XNNPACK delegates allocated in {load_time * 1000:.2f} ms")
        
        enhance_adapter = DataEnhanceAdapter()
        
        # Load and enhance images from selected animals
        print("Loading and enhancing depth images (DataEnhance)...")
        enhanced_imgs = []
        for t in selected_tags:
            frames = dataset.load_index(t)
            # Take up to 5 images per animal to benchmark a representative set without blowing RAM
            for f in frames[:5]:
                img = dataset.load_depth(t, f["depth_filename"])
                enhanced = enhance_adapter.run(img)
                enhanced_imgs.append(enhanced)
                
        print(f"Prepared {len(enhanced_imgs)} enhanced images for prediction.")
        
        # Warmup
        inference_adapter.predict([enhanced_imgs[0]])
        
        print("Running single-image weight inferences...")
        latencies = []
        start_bench = time.perf_counter()
        for img in enhanced_imgs:
            start_inf = time.perf_counter()
            inference_adapter.predict([img])
            inf_time = time.perf_counter() - start_inf
            latencies.append(inf_time)
        total_bench_time = time.perf_counter() - start_bench
            
        latencies_ms = np.array(latencies) * 1000
        print("\n--- RESULTS (Single-Image Mode) ---")
        print(f"Total Predictions Run: {len(enhanced_imgs)}")
        print(f"Total Inference Time:  {total_bench_time:.2f} seconds")
        print(f"Average Latency:       {np.mean(latencies_ms):.2f} ms")
        print(f"Min Latency:           {np.min(latencies_ms):.2f} ms")
        print(f"Max Latency:           {np.max(latencies_ms):.2f} ms")
        print(f"Std Dev:               {np.std(latencies_ms):.2f} ms")
        
        # Also benchmark batch prediction
        if len(enhanced_imgs) > 1:
            print(f"\nRunning batch inference benchmark on all {len(enhanced_imgs)} images at once...")
            start_batch = time.perf_counter()
            inference_adapter.predict(enhanced_imgs)
            batch_time = (time.perf_counter() - start_batch) * 1000
            print(f"Batch prediction time: {batch_time:.2f} ms (Avg per image in batch: {batch_time/len(enhanced_imgs):.2f} ms)")
            
        # Explicitly release memory
        del inference_adapter
        del enhanced_imgs
        gc.collect()
        print("Prediction model and enhanced image memory released.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Selection and Inference models on tags.")
    parser.add_argument("--tag", type=str, default=None, help="Benchmark a single specific tag")
    parser.add_argument("--num-animals", type=int, default=None, help="Benchmark on N random tags (default: 20)")
    parser.add_argument("--model", type=str, choices=["selection", "prediction", "all"], default="all", help="Model to benchmark")
    args = parser.parse_args()
    
    # If no parameters set, default to 20 random animals
    num_animals = args.num_animals
    if args.tag is None and num_animals is None:
        num_animals = 20
        
    run_benchmark(args.tag, num_animals, args.model)
