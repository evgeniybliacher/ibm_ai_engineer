import { HfInference } from "@huggingface/inference";
import { readFileSync } from "fs";
//const pdf = "pdf-parse";
import pdf from "pdf-parse";

const hf = new HfInference("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");

const extractTextFromPDF = async (filePath) => {
  try {
    const dataBuffer = readFileSync(filePath);
    const data = await pdf(dataBuffer);
    const text = data.text.replace(/\n/g, " ").replace(/ +/g, " ");
    return text;
  } catch (err) {
    console.error("Error extracting text from PDF:", err);
    throw err;
  }
};
const convertTextToEmbedding = async (text) => {
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
const filePath = "foodMenu.pdf";
async function main() {
  const text = await extractTextFromPDF(filePath);
  console.log("Extracted Text:", text);
  const embeddings = await convertTextToEmbedding(text);
  console.log(embeddings);
}
main();
