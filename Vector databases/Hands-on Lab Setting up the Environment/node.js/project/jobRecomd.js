const { ChromaClient } = require("chromadb");
const client = new ChromaClient();
const { HfInference } = require("@huggingface/inference");
const hf = new HfInference("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");

const jobPostings = require("./jobPostings.js");
const collectionName = "job_collection";

async function generateEmbeddings(texts) {
  const results = await hf.featureExtraction({
    model: "sentence-transformers/all-MiniLM-L6-v2",
    inputs: texts,
  });
  return results;
}

async function classifyText(text, labels) {
  const responses = await hf.zeroShotClassification({
    model: "facebook/bart-large-mnli",
    inputs: [text], // Must be an array
    parameters: {
      candidate_labels: labels,
    },
  });

  // Since we only passed one input, return the first response
  return responses[0];
}

async function extractFilterCriteria(query) {
  const criteria = {
    location: null,
    jobTitle: null,
    company: null,
    jobType: null,
  };
  const labels = ["location", "job title", "company", "job type"];

  const words = query.split(" ");
  for (const word of words) {
    const result = await classifyText(word, labels);
    console.log("result", result);
    const highestScoreLabel = result.labels[0];
    const score = result.scores[0];

    if (score > 0.5) {
      switch (highestScoreLabel) {
        case "location":
          criteria.location = word;
          break;
        case "job title":
          criteria.jobTitle = word;
          break;
        case "company":
          criteria.company = word;
          break;
        case "job type":
          criteria.jobType = word;
          break;
        default:
          break;
      }
    }
  }
  return criteria;
}

async function performSimilaritySearch(collection, queryTerm, filterCriteria) {
  try {
    const queryEmbedding = await generateEmbeddings([queryTerm]);
    console.log(filterCriteria);
    const results = await collection.query({
      collection: collectionName,
      queryEmbeddings: queryEmbedding,
      n: 3,
    });

    if (!results || results.length === 0) {
      console.log(`No job items found similar to "${queryTerm}"`);
      return [];
    }

    let topJobPostings = results.ids[0]
      .map((id, index) => {
        return {
          id,
          score: results.distances[0][index],
          job_title: jobPostings.find((item) => item.jobId.toString() === id)
            .jobTitle,
          food_description: jobPostings.find(
            (item) => item.jobId.toString() === id
          ).jobDescription,
        };
      })
      .filter(Boolean);
    return topJobPostings.sort((a, b) => a.score - b.score);
  } catch (error) {
    console.error("Error during similarity search:", error);
    return [];
  }
}

async function performSimilaritySearch(collection, queryTerm, filterCriteria) {
  try {
    const queryEmbedding = await generateEmbeddings([queryTerm]);
    console.log(filterCriteria);

    const results = await collection.query({
      queryEmbeddings: queryEmbedding,
      n: 3,
    });

    if (!results || results.ids[0].length === 0) {
      console.log(`No jobs found similar to "${queryTerm}"`);
      return [];
    }

    let topJobPostings = results.ids[0]
      .map((id, index) => {
        const jobs = jobPostings.find((item) => item.jobId.toString() === id);
        return {
          id,
          score: results.distances[0][index],
          job_title: jobs.jobTitle,
          job_description: jobs.jobDescription,
          job_type: jobs.jobType,
          job_company: jobs.company,
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
  const query = "Creative Studio";
  try {
    await client.deleteCollection({ name: collectionName });
    const collection = await client.getOrCreateCollection({
      name: collectionName,
    });
    const uniqueIds = new Set();
    jobPostings.forEach(async (job, index) => {
      while (uniqueIds.has(job.jobId.toString())) {
        job.jobId = `${job.jobId}_${index}`;
      }
      uniqueIds.add(job.jobId.toString());
    });
    const jobTexts = jobPostings.map(
      (job) =>
        `${job.jobTitle}, ${job.jobDescription}, ${job.jobType}, ${job.location}`
    );
    const embeddingsData = await generateEmbeddings(jobTexts);
    await collection.add({
      ids: jobPostings.map((job) => job.jobId.toString()),
      documents: jobTexts,
      embeddings: embeddingsData,
    });
    const filterCriteria = await extractFilterCriteria(query);
    const initialResults = await performSimilaritySearch(
      collection,
      query,
      filterCriteria
    );

    initialResults.slice(0, 3).forEach((item, index) => {
      console.log(
        `Top ${index + 1} Recommended Jobs ==>, ${item.job_title} ${
          item.job_type
        }, ${item.job_description} ${item.job_company}`
      );
    });
  } catch (error) {
    console.error("Error:", error);
  }
}

main();
