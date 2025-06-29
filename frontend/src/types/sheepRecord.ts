/**
 * Interface for the response received from the backend's /predict endpoint (multi-class).
 */
export interface SheepAnalysisResponse {
  filename: string;
  prediction: string;
  pain_probability: number;
  confidence: number;
  record_id?: string;
}

/**
 * Interface for a sheep record retrieved from the backend's /records endpoint.
 * This reflects the data structure stored in MongoDB.
 */
export interface SheepRecord {
  id: string;
  timestamp: Date;
  filename: string;
  prediction: string;
  confidence: number;
  device_id: string;
  pain_probability?: number;
}
