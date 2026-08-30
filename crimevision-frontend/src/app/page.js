'use client';

import React, { useState, useRef } from "react";

export default function Home() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");
  const [results, setResults] = useState(null);
  const [filterTab, setFilterTab] = useState("all"); // "all", "threats", "people", "vehicles"
  const [searchQuery, setSearchQuery] = useState("");
  const [llmReport, setLlmReport] = useState("");
  const [generatingReport, setGeneratingReport] = useState(false);
  
  const fileInputRef = useRef(null);

  // Status message rotation to simulate analysis pipeline steps
  const statusSteps = [
    "Uploading video file...",
    "Initializing YOLO Video Intelligence Engine...",
    "Extracting frames and tracking objects (ByteTrack)...",
    "Running spatial heuristics for attributes & color classification...",
    "Aggregating temporal track modes to reduce detection noise...",
    "Generating structured JSON analytics report..."
  ];

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const triggerUpload = () => {
    fileInputRef.current.click();
  };

  const handleUploadSubmit = async () => {
    if (!file) return;

    setLoading(true);
    setResults(null);
    setLlmReport("");
    
    // Rotate status messages every few seconds to show pipeline progression
    let stepIdx = 0;
    setStatusMsg(statusSteps[0]);
    const interval = setInterval(() => {
      if (stepIdx < statusSteps.length - 1) {
        stepIdx++;
        setStatusMsg(statusSteps[stepIdx]);
      }
    }, 3000);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8080/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Video processing failed.");
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      clearInterval(interval);
      setLoading(false);
      setStatusMsg("");
    }
  };

  // Generate a mock LLM incident report using the detection data
  const generateLLMReport = () => {
    if (!results) return;

    setGeneratingReport(true);
    setTimeout(() => {
      const threats = results.detections.filter(d => 
        ["Knife", "Gun", "Fire", "Smoke"].includes(d.object)
      );
      const persons = results.detections.filter(d => d.object === "Person");
      const vehicles = results.detections.filter(d => d.object === "Vehicle");

      let report = `### CRIMEVISION AI - AUTOMATED INCIDENT SECURITY REPORT\n`;
      report += `**Reference File**: ${results.video_name}\n`;
      report += `**Video Duration**: ${results.duration}\n`;
      report += `**Classification**: CONFIDENTIAL // FOR INTERNAL SURVEILLANCE REVIEW\n\n`;
      
      report += `#### 1. INCIDENT OVERVIEW\n`;
      if (threats.length > 0) {
        report += `⚠️ **HIGH ALERT**: Surveillance analytics detected potential security threats or environmental hazards in this video file. A total of ${threats.length} alarm event(s) were flagged.\n`;
      } else {
        report += `✅ **NORMAL RESOLUTION**: No active weapon threats or fire hazards detected. Routine surveillance recorded standard person/vehicle activities.\n`;
      }
      report += `The video contains records for **${persons.length} distinct person track(s)** and **${vehicles.length} distinct vehicle track(s)**.\n\n`;

      report += `#### 2. TIMELINE ANALYSIS\n`;
      results.detections.forEach(d => {
        let line = `- **[${d.timestamp}]**: Detected **${d.object}** (Track ID #${d.track_id}, Conf: ${(d.confidence * 100).toFixed(0)}%).`;
        if (d.shirt_color && d.shirt_color !== "Unknown") {
          line += ` Subject description: ${d.shirt_color} shirt, ${d.pant_color || "Unknown"} pants.`;
        }
        if (d.vehicle_color && d.vehicle_color !== "Unknown") {
          line += ` Vehicle description: ${d.vehicle_color} body color.`;
        }
        report += line + "\n";
      });
      report += `\n`;

      report += `#### 3. RISKS & ACTIONS SUMMARY\n`;
      if (threats.length > 0) {
        report += `🚨 **Threat Index**: HIGH RISK\n`;
        report += `**Flagged Items**: ${threats.map(t => t.object).join(", ")}\n`;
        report += `**Recommended Actions**:\n`;
        report += `1. Dispatch security personnel immediately to the physical sector corresponding to this camera feed.\n`;
        report += `2. Cross-reference identified Person track IDs with building access cards registered during the incident timeframe.\n`;
        report += `3. Archive this footage and log JSON timeline metadata to the central security database.\n`;
      } else {
        report += `🟢 **Threat Index**: LOW / ROUTINE MONITORING\n`;
        report += `No immediate tactical actions required. Standard logs updated.\n`;
      }

      setLlmReport(report);
      setGeneratingReport(false);
    }, 1500);
  };

  // Filter Detections based on Tab and Search Query
  const getFilteredDetections = () => {
    if (!results) return [];
    
    return results.detections.filter(d => {
      // 1. Tab filter
      if (filterTab === "threats" && !["Knife", "Gun", "Fire", "Smoke"].includes(d.object)) {
        return false;
      }
      if (filterTab === "people" && d.object !== "Person") {
        return false;
      }
      if (filterTab === "vehicles" && d.object !== "Vehicle") {
        return false;
      }

      // 2. Search query filter
      if (searchQuery.trim() !== "") {
        const query = searchQuery.toLowerCase();
        const objName = d.object.toLowerCase();
        const details = `${d.shirt_color || ""} ${d.pant_color || ""} ${d.vehicle_color || ""} #${d.track_id}`.toLowerCase();
        return objName.includes(query) || details.includes(query);
      }

      return true;
    });
  };

  // Count Categories
  const getStats = () => {
    if (!results) return { persons: 0, vehicles: 0, weapons: 0, hazards: 0 };
    let persons = 0, vehicles = 0, weapons = 0, hazards = 0;
    
    results.detections.forEach(d => {
      const obj = d.object.toLowerCase();
      if (obj === "person") persons++;
      else if (obj === "vehicle") vehicles++;
      else if (["knife", "gun"].includes(obj)) weapons++;
      else if (["fire", "smoke"].includes(obj)) hazards++;
    });

    return { persons, vehicles, weapons, hazards };
  };

  const stats = getStats();
  const filteredDetections = getFilteredDetections();

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-emerald-500/30">
      {/* 1. Header */}
      <header className="border-b border-slate-900 bg-slate-950/70 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-4 h-4 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_10px_#10b981]" />
          <div>
            <h1 className="text-xl font-bold tracking-wider text-slate-100 flex items-center gap-2">
              CRIMEVISION <span className="text-emerald-500 text-xs font-mono px-2 py-0.5 bg-emerald-500/10 border border-emerald-500/20 rounded">YOLO ENGINE</span>
            </h1>
            <p className="text-[10px] text-slate-500 tracking-widest font-mono">VIDEO INTELLIGENCE TERMINAL</p>
          </div>
        </div>
        <div className="flex items-center gap-4 text-xs font-mono text-slate-400">
          <div>FEED STATUS: <span className="text-emerald-500 font-bold">ONLINE</span></div>
        </div>
      </header>

      {/* 2. Main Content */}
      <main className="flex-1 p-6 flex flex-col lg:flex-row gap-6 max-w-7xl mx-auto w-full">
        {/* Upload State / Left side Video player */}
        <div className="flex-1 flex flex-col gap-6">
          {!results && !loading && (
            <div 
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              className="border-2 border-dashed border-slate-800 hover:border-emerald-500/50 bg-slate-900/30 transition-all rounded-xl p-12 flex flex-col items-center justify-center text-center gap-4 cursor-pointer min-h-[400px] shadow-[inset_0_4px_30px_rgba(0,0,0,0.4)]"
            >
              <div className="w-16 h-16 rounded-full bg-slate-900 flex items-center justify-center border border-slate-800 text-slate-400">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                </svg>
              </div>
              <div>
                <h3 className="text-lg font-semibold text-slate-200">Upload Video Feed</h3>
                <p className="text-sm text-slate-500 mt-1 max-w-xs mx-auto">
                  Drag and drop your surveillance video files (.mp4, .avi, .mov) or click below to browse.
                </p>
              </div>
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange}
                accept="video/*"
                className="hidden"
              />
              <button 
                onClick={triggerUpload}
                className="px-5 py-2.5 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg text-sm text-slate-300 font-medium transition-all"
              >
                Browse Files
              </button>

              {file && (
                <div className="mt-4 p-3 bg-slate-900/90 border border-slate-800 rounded-lg max-w-sm flex items-center justify-between w-full">
                  <div className="flex items-center gap-2 overflow-hidden">
                    <span className="text-emerald-500 font-mono text-xs">✓</span>
                    <span className="text-xs text-slate-300 truncate font-mono">{file.name}</span>
                  </div>
                  <button 
                    onClick={handleUploadSubmit}
                    className="ml-3 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-emerald-950 font-bold rounded text-xs transition-all"
                  >
                    Analyze
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Loading State */}
          {loading && (
            <div className="border border-slate-900 bg-slate-900/10 rounded-xl p-12 flex flex-col items-center justify-center min-h-[400px] gap-6 text-center relative overflow-hidden">
              {/* Radar scanner sweep effect */}
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(16,185,129,0.03)_0%,transparent_70%)] animate-pulse" />
              <div className="relative">
                <div className="w-20 h-20 rounded-full border-2 border-emerald-500/20 border-t-emerald-500 animate-spin" />
                <div className="absolute inset-0 flex items-center justify-center text-[10px] font-mono text-emerald-500">YOLO</div>
              </div>
              <div className="max-w-sm relative z-10">
                <h3 className="text-lg font-semibold text-slate-200">Analyzing Feed</h3>
                <p className="text-xs text-emerald-500 font-mono mt-2 h-4 animate-pulse">{statusMsg}</p>
                <div className="w-48 h-1 bg-slate-950 rounded-full mx-auto mt-4 overflow-hidden border border-slate-900">
                  <div className="h-full bg-emerald-500 rounded-full animate-[loading_2s_infinite]" />
                </div>
              </div>
            </div>
          )}

          {/* Analysis Video Preview Screen */}
          {results && (
            <div className="flex flex-col gap-6">
              <div className="border border-slate-900 bg-slate-900/20 rounded-xl p-1.5 shadow-2xl relative overflow-hidden">
                <video 
                  src={results.annotated_video_url} 
                  controls 
                  autoPlay
                  className="w-full rounded-lg aspect-video bg-slate-950 border border-slate-950"
                />
              </div>

              {/* Grid Widgets (Counts) */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-4 bg-slate-900/20 border border-slate-900/60 rounded-xl flex flex-col gap-1">
                  <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">Duration</span>
                  <span className="text-xl font-bold font-mono text-slate-200">{results.duration}</span>
                </div>
                <div className="p-4 bg-slate-900/20 border border-slate-900/60 rounded-xl flex flex-col gap-1">
                  <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">People Logged</span>
                  <span className="text-xl font-bold font-mono text-emerald-500">{stats.persons}</span>
                </div>
                <div className="p-4 bg-slate-900/20 border border-slate-900/60 rounded-xl flex flex-col gap-1">
                  <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">Vehicles Logged</span>
                  <span className="text-xl font-bold font-mono text-blue-500">{stats.vehicles}</span>
                </div>
                <div className="p-4 bg-slate-900/20 border border-slate-900/60 rounded-xl flex flex-col gap-1 relative overflow-hidden">
                  <div className={`absolute top-0 right-0 w-16 h-16 bg-gradient-to-br ${stats.weapons > 0 ? "from-red-500/10 to-transparent" : "from-orange-500/10 to-transparent"}`} />
                  <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">Threats / Hazards</span>
                  <span className={`text-xl font-bold font-mono ${stats.weapons > 0 ? "text-red-500 animate-pulse shadow-red-500/20" : stats.hazards > 0 ? "text-orange-500" : "text-slate-400"}`}>
                    {stats.weapons + stats.hazards}
                  </span>
                </div>
              </div>

              {/* Quick Reset */}
              <button 
                onClick={() => { setFile(null); setResults(null); }}
                className="self-start text-xs font-mono text-slate-500 hover:text-slate-300 transition-all flex items-center gap-1.5"
              >
                ← Analyze Another Video
              </button>
            </div>
          )}
        </div>

        {/* Right side Log & LLM analysis */}
        <div className="w-full lg:w-[480px] flex flex-col gap-6">
          {/* Controls Table */}
          {results ? (
            <div className="border border-slate-900 bg-slate-900/30 rounded-xl p-5 flex flex-col gap-4 shadow-xl flex-1 max-h-[640px] overflow-hidden">
              <div className="flex items-center justify-between pb-3 border-b border-slate-900">
                <h3 className="text-sm font-bold tracking-wider font-mono text-slate-400">SURVEILLANCE EVENT LOG</h3>
                <span className="px-2 py-0.5 rounded bg-slate-800 text-[10px] text-slate-400 font-mono">
                  {filteredDetections.length} Items
                </span>
              </div>

              {/* Search Box */}
              <div className="relative">
                <input 
                  type="text" 
                  placeholder="Filter by object or attribute (e.g. Red, Knife)..." 
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-950 border border-slate-900 rounded-lg text-xs placeholder:text-slate-600 focus:outline-none focus:border-emerald-500/50 transition-all font-mono"
                />
              </div>

              {/* Tabs */}
              <div className="grid grid-cols-4 gap-1.5 p-1 bg-slate-950 rounded-lg border border-slate-900">
                {["all", "threats", "people", "vehicles"].map(t => (
                  <button
                    key={t}
                    onClick={() => setFilterTab(t)}
                    className={`py-1.5 rounded text-[10px] font-mono capitalize transition-all ${filterTab === t ? "bg-slate-900 border border-slate-800 text-slate-100" : "text-slate-500 hover:text-slate-300"}`}
                  >
                    {t}
                  </button>
                ))}
              </div>

              {/* Timeline Items */}
              <div className="flex-1 overflow-y-auto pr-1 flex flex-col gap-2.5">
                {filteredDetections.length > 0 ? (
                  filteredDetections.map((d, index) => {
                    const isWeapon = ["Knife", "Gun"].includes(d.object);
                    const isHazard = ["Fire", "Smoke"].includes(d.object);
                    
                    return (
                      <div 
                        key={index} 
                        className={`p-3 bg-slate-950/70 border rounded-lg flex items-start justify-between gap-3 transition-all ${isWeapon ? "border-red-950 hover:border-red-900/50 bg-red-950/5" : isHazard ? "border-orange-950 hover:border-orange-900/50 bg-orange-950/5" : "border-slate-900 hover:border-slate-800"}`}
                      >
                        <div className="flex flex-col gap-1">
                          <div className="flex items-center gap-2">
                            <span className={`px-2 py-0.5 rounded font-mono text-[9px] font-bold tracking-wider ${isWeapon ? "bg-red-500/10 text-red-400 border border-red-500/20 shadow-[0_0_5px_rgba(239,68,68,0.1)]" : isHazard ? "bg-orange-500/10 text-orange-400 border border-orange-500/20" : d.object === "Person" ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" : "bg-blue-500/10 text-blue-400 border border-blue-500/20"}`}>
                              {d.object}
                            </span>
                            <span className="text-[10px] text-slate-500 font-mono">ID #{d.track_id}</span>
                          </div>
                          
                          {/* Attribute details */}
                          {d.object === "Person" && (
                            <div className="text-[11px] text-slate-400 font-mono mt-0.5 flex gap-2">
                              <span>Shirt: <b className="text-slate-300 font-bold">{d.shirt_color}</b></span>
                              <span>Pants: <b className="text-slate-300 font-bold">{d.pant_color}</b></span>
                            </div>
                          )}
                          {d.object === "Vehicle" && (
                            <div className="text-[11px] text-slate-400 font-mono mt-0.5">
                              <span>Body Color: <b className="text-slate-300 font-bold">{d.vehicle_color}</b></span>
                            </div>
                          )}
                        </div>
                        <div className="flex flex-col items-end gap-1 font-mono text-[10px]">
                          <span className="text-emerald-500 font-bold">{d.timestamp}</span>
                          <span className="text-slate-600">{(d.confidence * 100).toFixed(0)}% Conf</span>
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <div className="flex-1 flex flex-col items-center justify-center p-8 text-center text-slate-600 font-mono text-xs">
                    No detections match current filter.
                  </div>
                )}
              </div>

              {/* Interactive AI report */}
              <div className="border-t border-slate-900 pt-4 flex flex-col gap-3">
                {!llmReport && (
                  <button
                    onClick={generateLLMReport}
                    disabled={generatingReport}
                    className="w-full py-2 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-emerald-950 font-bold rounded-lg text-xs font-mono tracking-wider transition-all disabled:opacity-50"
                  >
                    {generatingReport ? "COMPILING ANALYSIS DATA..." : "GENERATE AI INCIDENT REPORT"}
                  </button>
                )}
                
                {llmReport && (
                  <div className="bg-slate-950 border border-slate-900 rounded-lg p-4 font-mono text-[10px] text-slate-300 max-h-[220px] overflow-y-auto relative flex flex-col gap-2">
                    <div className="text-xs font-bold text-emerald-500 border-b border-slate-900 pb-1.5 flex items-center justify-between sticky top-0 bg-slate-950">
                      <span>✓ REPORT GENERATED</span>
                      <button 
                        onClick={() => setLlmReport("")}
                        className="text-slate-500 hover:text-slate-300 text-[10px]"
                      >
                        [Clear]
                      </button>
                    </div>
                    <pre className="whitespace-pre-wrap leading-relaxed select-all pr-1">{llmReport}</pre>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="border border-slate-900 bg-slate-900/20 rounded-xl p-6 flex flex-col items-center justify-center text-center gap-2 shadow-xl min-h-[300px]">
              <span className="text-xs font-mono text-slate-600 tracking-wider">AWAITING VIDEO INGESTION</span>
              <p className="text-[10px] text-slate-700 max-w-[200px]">
                Once a video is uploaded and analyzed, interactive timeline logs and AI report compilers will activate here.
              </p>
            </div>
          )}
        </div>
      </main>

      {/* 3. Footer */}
      <footer className="mt-auto border-t border-slate-900 px-6 py-4 bg-slate-950 text-center font-mono text-[10px] text-slate-600">
        CRIMEVISION SURVEILLANCE SUITE • POWERED BY ULTRALYTICS YOLO & BYTETRACK
      </footer>

      {/* Tailwind helper style animation */}
      <style jsx global>{`
        @keyframes loading {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </div>
  );
}
