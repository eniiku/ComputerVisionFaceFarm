import { API_CONFIG } from "@/config/env";

export interface SheepAnalysisResult {
  filename: string;
  prediction: string;
  pain_probability: number;
  confidence: number;
}

export class SheepAnalysisApi {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_CONFIG.BASE_URL;
  }

  async analyzeSheepImage(imageFile: File): Promise<SheepAnalysisResult> {
    const formData = new FormData();
    formData.append("file", imageFile);

    const response = await fetch(`${this.baseUrl}/predict`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(
        `API request failed: ${response.status} ${response.statusText}`
      );
    }

    return await response.json();
  }
}

export const sheepAnalysisApi = new SheepAnalysisApi();
