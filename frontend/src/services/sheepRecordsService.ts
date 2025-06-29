import { API_CONFIG } from "@/config/env";
import { SheepRecord } from "@/types/sheepRecord";

import { sheepAnalysisApi } from "./sheepAnalysisApi";

export class SheepRecordsService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_CONFIG.BASE_URL;
  }

  /**
   * Fetches all sheep pain prediction records for the current device from the backend.
   * Records are ordered from most recent to oldest.
   */
  async getRecords(): Promise<SheepRecord[]> {
    console.log("Fetching sheep records from backend...");

    const deviceId = sheepAnalysisApi.getDeviceId();

    if (!deviceId) {
      console.warn("No device ID found. Cannot fetch records.");
      return [];
    }

    try {
      const response = await fetch(`${this.baseUrl}/records`, {
        method: "GET",
        headers: {
          "X-Device-ID": deviceId,
        },
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("API response error:", errorText);
        throw new Error(
          `Failed to fetch records: ${response.status} ${response.statusText}. Detail: ${errorText}`
        );
      }

      const data = await response.json();
      console.log(
        `Retrieved ${data.records.length} records for device ID: ${deviceId}`
      );

      return data.records.map((record: any) => ({
        ...record,
        timestamp: new Date(record.timestamp),
      }));
    } catch (error) {
      console.error("Error fetching records:", error);
      return [];
    }
  }
}

export const sheepRecordsService = new SheepRecordsService();
