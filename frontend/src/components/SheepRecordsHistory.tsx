import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { AlertTriangle, CheckCircle, History, ArrowLeft } from "lucide-react";
import { SheepRecord } from "@/types/sheepRecord";
import { sheepRecordsService } from "@/services/sheepRecordsService";

interface SheepRecordsHistoryProps {
  onBack: () => void;
}

const SheepRecordsHistory = ({ onBack }: SheepRecordsHistoryProps) => {
  const [records, setRecords] = useState<SheepRecord[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadRecords();
  }, []);

  const loadRecords = async () => {
    try {
      const fetchedRecords = await sheepRecordsService.getRecords();
      setRecords(fetchedRecords);
    } catch (error) {
      console.error("Failed to load records:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  };

  const translatePrediction = (rawPrediction: string): string => {
    switch (rawPrediction) {
      case "corpus_sheep_face_pain":
        return "Pain Detected";
      case "corpus_sheep_face_no_pain":
        return "Healthy";
      default:
        return rawPrediction; // fallback to raw value if unknown
    }
  };

  if (isLoading) {
    return (
      <Card className="bg-black/40 backdrop-blur-xl border-purple-500/30 shadow-2xl">
        <CardContent className="p-6">
          <div className="text-center">
            <History className="h-12 w-12 text-purple-400 animate-pulse mx-auto mb-4" />
            <p className="text-purple-300">Loading records...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-black/40 backdrop-blur-xl border-purple-500/30 shadow-2xl">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <Button
            variant="ghost"
            size="sm"
            onClick={onBack}
            className="text-purple-300 hover:text-white hover:bg-purple-500/20"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back
          </Button>
          <CardTitle className="text-white flex items-center space-x-2">
            <History className="h-5 w-5 text-purple-400" />
            <span>Sheep Analysis History</span>
          </CardTitle>
          <div className="w-16"></div>
        </div>
      </CardHeader>
      <CardContent>
        {records.length === 0 ? (
          <div className="text-center py-8">
            <History className="h-16 w-16 text-purple-400/50 mx-auto mb-4" />
            <p className="text-purple-300 text-lg mb-2">No records yet</p>
            <p className="text-purple-400 text-sm">
              Analyze some sheep to see their records here
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            <p className="text-purple-300 text-sm">
              Total records:{" "}
              <span className="font-semibold text-white">{records.length}</span>
            </p>

            <div className="overflow-x-auto">
              <Table className="text-white">
                <TableHeader>
                  <TableRow className="border-purple-500/30">
                    <TableHead className="text-purple-300">Image</TableHead>
                    <TableHead className="text-purple-300">Status</TableHead>
                    <TableHead className="text-purple-300">
                      Pain Probability
                    </TableHead>
                    <TableHead className="text-purple-300">
                      Confidence
                    </TableHead>
                    <TableHead className="text-purple-300">Date</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {records.map((record) => (
                    <TableRow
                      key={record.id}
                      className="border-purple-500/20 hover:bg-purple-500/10"
                    >
                      <TableCell>
                        <div className="flex items-center space-x-3">
                          <span className="text-sm text-purple-300 truncate max-w-[100px]">
                            {record.filename}
                          </span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge
                          className={`${
                            record.pain_probability > 0.5
                              ? "bg-gradient-to-r from-red-600 to-red-500"
                              : "bg-gradient-to-r from-green-600 to-green-500"
                          } text-white`}
                        >
                          <div className="flex items-center space-x-1">
                            {record.pain_probability > 0.5 ? (
                              <AlertTriangle className="h-3 w-3" />
                            ) : (
                              <CheckCircle className="h-3 w-3" />
                            )}
                            <span>
                              {translatePrediction(record.prediction)}
                            </span>
                          </div>
                        </Badge>
                      </TableCell>
                      <TableCell className="text-purple-300">
                        {(record.pain_probability * 100).toFixed(1)}%
                      </TableCell>
                      <TableCell className="text-purple-300">
                        {(record.confidence * 100).toFixed(1)}%
                      </TableCell>
                      <TableCell className="text-purple-300 text-sm">
                        {formatDate(record.timestamp)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default SheepRecordsHistory;
