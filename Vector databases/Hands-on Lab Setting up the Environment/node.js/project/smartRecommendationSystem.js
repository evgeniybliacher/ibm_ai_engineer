const { ChromaClient } = require("chromadb");
const client = new ChromaClient();
const { HfInference } = require("@huggingface/inference");
const hf = new HfInference("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");
const pdf = require("pdf-parse");
const jobPostings = require("./jobPostings.js");
const collectionName = "smart_job_collection";
const fs = require("fs");

const readline = require("readline");

// Create an interface for input and output
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const promptUserInput = (question) => {
  return new Promise((resolve) => {
    rl.question(question, (answer) => {
      resolve(answer);
    });
  });
};

const extractTextFromPDF = async (filePath) => {
  try {
    const dataBuffer = fs.readFileSync(filePath);
    const data = await pdf(dataBuffer);
    const text = data.text.replace(/\n/g, " ").replace(/ +/g, " ");
    return text;
  } catch (err) {
    console.error("Error extracting text from PDF:", err);
    throw err;
  }
};
const generateEmbeddings = async (text) => {
  try {
    const result = await hf.featureExtraction({
      model: "sentence-transformers/all-MiniLM-L6-v2",
      inputs: text,
    });
    // console.log("Embedding Result:", result);
    return result; // Return the embedding array
  } catch (err) {
    console.error("Error converting text to embeddings:", err);
    throw err;
  }
};

const storeEmbeddings = async (jobPostings) => {
  const jobEmbeddings = [];
  const metadatas = jobPostings.map(() => ({})); // Empty metadata objects
  for (const job of jobPostings) {
    const embedding = await generateEmbeddings(
      job.jobDescription.toLowerCase()
    );
    jobEmbeddings.push(embedding);
  }
  const ids = jobPostings.map((_, index) => index.toString());
  const jobTexts = jobPostings.map((job) => job.jobTitle);
  try {
    const collection = await client.getOrCreateCollection({
      name: collectionName,
    });

    await collection.add({
      ids: ids,
      documents: jobTexts,
      embeddings: jobEmbeddings,
      metadatas: metadatas,
    });
    console.log("Stored embeddings in Chroma DB.");
    return collection;
  } catch (error) {
    console.error("Error storing embeddings in Chroma DB:", error);
    throw error;
  }
};

async function performSimilaritySearch(collection, queryEmbedding) {
  try {
    const results = await collection.query({
      queryEmbeddings: queryEmbedding,
      n: 3,
    });

    if (!results || results.ids[0].length === 0) {
      console.log(`No jobs found similar to`);
      return [];
    }

    let topJobPostings = results.ids[0]
      .map((id, index) => {
        const jobs = jobPostings.find((item) => item.jobId.toString() === id);
        console.log(jobs.jobTitle);
        return {
          id,
          score: results.distances[0][index],
          job_title: jobs.jobTitle,
        };
      })
      .filter(Boolean);

    return topJobPostings.sort((a, b) => a.score - b.score);
  } catch (error) {
    console.error("Error during similarity search:", error);
    return [];
  }
}
async function main() {
  try {
    const collection = await storeEmbeddings(jobPostings);
    const resumeFilePath = await promptUserInput(
      "Enter the path to the candidat`s resume PDF: "
    );
    const text = await extractTextFromPDF(resumeFilePath);
    const resumeEmbedding = await generateEmbeddings(text);
    const results = await performSimilaritySearch(collection, resumeEmbedding);

    results.slice(0, 3).forEach((item, index) => {
      console.log(`Top ${index + 1} Recommended Jobs ==>, ${item.job_title}`);
    });
  } catch (error) {
    console.error("Error:", error);
  }
}

main();
