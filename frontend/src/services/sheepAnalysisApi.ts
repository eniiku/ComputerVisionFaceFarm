import { API_CONFIG } from "@/config/env";
import { v4 as uuidv4 } from "uuid";

import { SheepAnalysisResponse } from "@/types/sheepRecord";

export class SheepAnalysisApi {
  private baseUrl: string;
  private deviceId: string;

  constructor() {
    this.baseUrl = API_CONFIG.BASE_URL;
    this.deviceId = this.getOrCreateDeviceId();
  }

  private getOrCreateDeviceId(): string {
    let deviceId = localStorage.getItem("sheep_app_device_id");
    if (!deviceId) {
      deviceId = uuidv4();
      localStorage.setItem("sheep_app_device_id", deviceId);
      console.log("Generated new device ID:", deviceId);
    } else {
      console.log("Using existing device ID:", deviceId);
    }
    return deviceId;
  }

  public getDeviceId(): string {
    return this.deviceId;
  }

  async analyzeSheepImage(imageFile: File): Promise<SheepAnalysisResponse> {
    const formData = new FormData();
    formData.append("file", imageFile);

    console.log(`Sending image for analysis from device: ${this.deviceId}`);

    const response = await fetch(`${this.baseUrl}/predict`, {
      method: "POST",
      headers: {
        "X-Device-ID": this.deviceId,
      },
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API response error:", errorText);
      throw new Error(
        `API request failed: ${response.status} ${response.statusText}. Detail: ${errorText}`
      );
    }

    const result: SheepAnalysisResponse = await response.json();
    console.log("Analysis result received:", result);
    return result;
  }
}

export const sheepAnalysisApi = new SheepAnalysisApi();
