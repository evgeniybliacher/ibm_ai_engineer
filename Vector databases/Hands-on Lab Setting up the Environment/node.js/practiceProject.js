import { ChromaClient } from "chromadb"; // Imports the ChromaClient class from the chromadb library
import { HfInference } from "@huggingface/inference";

const hf = new HfInference("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");

const client = new ChromaClient(); // Creates a new ChromaClient instance
