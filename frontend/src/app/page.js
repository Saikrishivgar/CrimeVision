'use client';

import React, { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { Search, Play, Mic, Sparkles, Download, MapPin, Video, Camera, Clock, Activity, FileText, Settings, ShieldAlert, CheckCircle2, AlertTriangle, Send, User, ChevronRight, UploadCloud, X, Loader2, Bot, AlertCircle, MessageSquare, PlusCircle, Trash2 } from "lucide-react";
import axios from "axios";

// ─── EvidenceCard ───────────────────────────────────────────────
function EvidenceCard({ event, onAskWhy }) {
  const [evidence, setEvidence] = useState(null);
  const [loading, setLoading] = useState(true);
  const videoRef = useRef(null);

  useEffect(() => {
    axios.get(`http://localhost:8080/events/${event.event_id}/evidence`)
      .then(res => { setEvidence(res.data); })
      .catch(err => console.error("Error fetching evidence", err))
      .finally(() => setLoading(false));
  }, [event.event_id]);

  const handleOpenTimestamp = () => {
    if (videoRef.current) {
      const relTime = event.timestamp - (evidence?.clip_start || 0);
      videoRef.current.currentTime = Math.max(0, relTime);
      videoRef.current.play();
    }
  };

  const handlePlay = () => {
    if (videoRef.current) videoRef.current.play();
  };

  return (
    <div className="bg-[#1a2133] border border-blue-500/20 rounded-xl p-4 flex flex-col gap-3 shadow-lg">
      <div className="flex justify-between items-start">
        <div className="flex items-center gap-2">
          <span className="text-xl">
            {event.object.toLowerCase() === 'person' ? '🧍' : 
             ['car','truck','suv','bus'].includes(event.object.toLowerCase()) ? '🚗' : '🔍'}
          </span>
          <h3 className="font-bold text-white capitalize">
            {(event.color || event.attributes?.object_color || event.attributes?.vehicle_color) && (event.color || event.attributes?.object_color || event.attributes?.vehicle_color) !== "Unknown" ? (event.color || event.attributes?.object_color || event.attributes?.vehicle_color) + " " : ""}
            {event.object}
          </h3>
        </div>
        <div className="flex gap-2">
          {event.similarity && (
             <div className="bg-purple-500/10 text-purple-400 text-[10px] font-bold px-2 py-1 rounded-md border border-purple-500/20" title="AI Semantic Match Score">
               AI Match: {Math.round(event.similarity * 100)}%
             </div>
          )}
          {event.confidence && (
             <div className="bg-blue-500/10 text-blue-400 text-[10px] font-bold px-2 py-1 rounded-md border border-blue-500/20" title="YOLO Detection Confidence">
               Conf: {Math.round(event.confidence * 100)}%
             </div>
          )}
        </div>
      </div>
      
      <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden border border-white/10 group">
        {loading ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 gap-2">
             <Loader2 className="w-6 h-6 animate-spin" />
             <span className="text-xs">Extracting Evidence...</span>
          </div>
        ) : evidence ? (
          <video 
            ref={videoRef}
            src={`http://localhost:8080${evidence.clip_url}`}
            poster={`http://localhost:8080${evidence.frame_url}`}
            controls
            controlsList="nodownload"
            className="w-full h-full object-contain"
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center text-red-400 text-xs">
             Evidence Unavailable
          </div>
        )}
      </div>

      {evidence && (
        <div className="flex flex-col gap-1 mt-1">
          <div className="flex justify-between text-[10px] text-slate-400 font-mono">
            <span>{evidence.clip_start?.toFixed(1)}s</span>
            <span>{evidence.clip_end?.toFixed(1)}s</span>
          </div>
          <div className="relative w-full h-1 bg-slate-700 rounded-full">
            <div 
              className="absolute top-1/2 -translate-y-1/2 w-2 h-2 bg-blue-500 rounded-full shadow-[0_0_5px_#3b82f6]"
              style={{ 
                left: `${((event.timestamp - evidence.clip_start) / (evidence.clip_end - evidence.clip_start)) * 100}%` 
              }}
              title="Detection Timestamp"
            ></div>
          </div>
        </div>
      )}
      
      <div className="grid grid-cols-2 gap-y-2 gap-x-4 text-xs text-slate-300 bg-black/20 p-3 rounded-lg mt-1">
        <div className="flex items-center gap-2"><Video className="w-3 h-3 text-slate-500"/> {event.video_name}</div>
        <div className="flex items-center gap-2"><MapPin className="w-3 h-3 text-slate-500"/> {event.location}</div>
        <div className="flex items-center gap-2"><Clock className="w-3 h-3 text-slate-500"/> {event.timestamp}s</div>
        <div className="flex items-center gap-2"><Activity className="w-3 h-3 text-slate-500"/> {(event.source_model || "yolo11").toUpperCase()}</div>
        {event.stationary && <div className="flex items-center gap-2"><Clock className="w-3 h-3 text-slate-500"/> Stationary</div>}
      </div>
      
      <div className="flex gap-2 mt-2">
        <button onClick={handleOpenTimestamp} disabled={loading || !evidence} className="flex-1 bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 border border-blue-500/30 disabled:opacity-50 disabled:cursor-not-allowed rounded-md py-1.5 text-xs font-semibold flex items-center justify-center gap-1 transition-colors">
          <Clock className="w-3 h-3" /> Open Timestamp
        </button>
        <button onClick={handlePlay} disabled={loading || !evidence} className="flex-1 bg-white/5 hover:bg-white/10 border border-white/10 disabled:opacity-50 disabled:cursor-not-allowed rounded-md py-1.5 text-xs font-semibold flex items-center justify-center gap-1 transition-colors text-white">
          <Play className="w-3 h-3" /> Play Evidence
        </button>
        <button onClick={() => onAskWhy(event)} className="flex-1 bg-white/5 hover:bg-white/10 border border-white/10 rounded-md py-1.5 text-xs font-semibold flex items-center justify-center gap-1 transition-colors text-slate-300">
          <Search className="w-3 h-3" /> Why?
        </button>
      </div>
    </div>
  );
}

// ─── IncidentTimeline (Sidebar) ─────────────────────────────────
function IncidentTimeline({ videoId, onEventClick }) {
  const [timeline, setTimeline] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!videoId) return;
    setLoading(true);
    axios.get(`http://localhost:8080/incidents/timeline?video_id=${videoId}`)
      .then(res => setTimeline(res.data.timeline || []))
      .catch(err => console.error("Error fetching timeline:", err))
      .finally(() => setLoading(false));
  }, [videoId]);

  if (!videoId) return null;

  return (
    <div className="hidden md:flex flex-col w-full h-[50vh] mt-4 border border-white/10 bg-black/20 rounded-xl overflow-hidden">
      <div className="p-3 border-b border-white/10 bg-[#151b2b] flex items-center gap-2">
        <Activity className="w-4 h-4 text-emerald-400" />
        <h3 className="font-bold text-xs text-white uppercase tracking-wider">Incident Timeline</h3>
      </div>
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {loading ? (
          <div className="text-center text-xs text-slate-500 mt-4"><Loader2 className="w-4 h-4 animate-spin inline mr-2" />Loading timeline...</div>
        ) : timeline.length === 0 ? (
          <div className="text-center text-xs text-slate-500 mt-4">No events detected.</div>
        ) : (
          timeline.map((item, idx) => (
            <div 
              key={idx} 
              onClick={() => onEventClick(item)}
              className={`p-2 rounded-lg border text-xs cursor-pointer hover:bg-white/5 transition-colors ${
                item.type === 'incident' ? 'border-red-500/30 bg-red-500/5' : 'border-white/5 bg-[#151b2b]'
              }`}
            >
              <div className="flex justify-between items-start mb-1">
                <span className={`font-bold ${item.type === 'incident' ? 'text-red-400' : 'text-blue-400'}`}>
                  {item.timestamp}s
                </span>
                {item.confidence && (
                  <span className="text-[9px] text-slate-400">
                    {Math.round(item.confidence * 100)}% conf
                  </span>
                )}
              </div>
              <div className="text-slate-300 font-semibold mb-1 capitalize">
                {item.type === 'incident' ? item.event_type.replace('_', ' ') : 'Object Detected'}
              </div>
              <div className="text-[10px] text-slate-400 leading-tight">
                {item.description}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

// ─── ResultList ─────────────────────────────────────────────────
function ResultList({ results, onAskWhy }) {
  const [expanded, setExpanded] = useState(false);

  const groups = {};
  results.forEach(res => {
    const key = res.track_uid || ((res.track_id !== undefined && res.track_id !== null && res.track_id !== -1) ? `${res.video_id}_${res.track_id}` : res.event_id);
    if (!groups[key]) {
      groups[key] = res;
    } else {
      if (res.confidence > groups[key].confidence) {
        groups[key] = res;
      }
    }
  });

  const uniqueResults = Object.values(groups).sort((a, b) => b.confidence - a.confidence);
  const displayResults = expanded ? uniqueResults : uniqueResults.slice(0, 3);

  return (
    <div className="flex flex-col gap-4 mt-2">
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        {displayResults.map((res, ridx) => (
          <EvidenceCard key={ridx} event={res} onAskWhy={onAskWhy} />
        ))}
      </div>
      {uniqueResults.length > 3 && !expanded && (
        <button 
          onClick={() => setExpanded(true)}
          className="w-full py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-sm font-semibold transition-colors"
        >
          Show all {uniqueResults.length} matches
        </button>
      )}
    </div>
  );
}

// ─── MapCanvas (Canvas-based 2D Trajectory Map) ─────────────────
function MapCanvas({ tracks, incidents, highlightedTrackId, showTrails, showLabels, showIncidents, getTrackColor }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !tracks || tracks.length === 0) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    tracks.forEach(t => {
      (t.points || []).forEach(p => {
        const cx = (p.bbox[0] + p.bbox[2]) / 2;
        const cy = (p.bbox[1] + p.bbox[3]) / 2;
        if (cx < minX) minX = cx;
        if (cy < minY) minY = cy;
        if (cx > maxX) maxX = cx;
        if (cy > maxY) maxY = cy;
      });
    });
    const padX = (maxX - minX) * 0.1 || 50;
    const padY = (maxY - minY) * 0.1 || 50;
    minX -= padX; minY -= padY; maxX += padX; maxY += padY;
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const scale = Math.min(W / rangeX, H / rangeY);
    const offsetX = (W - rangeX * scale) / 2;
    const offsetY = (H - rangeY * scale) / 2;
    const toCanvas = (px, py) => [offsetX + (px - minX) * scale, offsetY + (py - minY) * scale];

    ctx.fillStyle = "#05070a";
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (let y = 0; y < H; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

    if (showTrails) {
      tracks.forEach(t => {
        if (!t.points || t.points.length < 2) return;
        const color = getTrackColor(t.track_id);
        const isHL = highlightedTrackId === t.track_id;
        ctx.strokeStyle = color;
        ctx.lineWidth = isHL ? 4 : 2;
        ctx.globalAlpha = isHL ? 1.0 : 0.6;
        ctx.beginPath();
        const pts = t.points;
        const [sx, sy] = toCanvas((pts[0].bbox[0] + pts[0].bbox[2]) / 2, (pts[0].bbox[1] + pts[0].bbox[3]) / 2);
        ctx.moveTo(sx, sy);
        for (let i = 1; i < pts.length; i++) {
          const [px, py] = toCanvas((pts[i].bbox[0] + pts[i].bbox[2]) / 2, (pts[i].bbox[1] + pts[i].bbox[3]) / 2);
          ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.globalAlpha = 1.0;

        const [startX, startY] = toCanvas((pts[0].bbox[0] + pts[0].bbox[2]) / 2, (pts[0].bbox[1] + pts[0].bbox[3]) / 2);
        ctx.fillStyle = "#10b981";
        ctx.beginPath(); ctx.arc(startX, startY, 5, 0, Math.PI * 2); ctx.fill();

        const last = pts[pts.length - 1];
        const [endX, endY] = toCanvas((last.bbox[0] + last.bbox[2]) / 2, (last.bbox[1] + last.bbox[3]) / 2);
        ctx.fillStyle = "#ef4444";
        ctx.beginPath(); ctx.arc(endX, endY, 5, 0, Math.PI * 2); ctx.fill();

        if (pts.length > 5) {
          const mid = Math.floor(pts.length / 2);
          const [mx, my] = toCanvas((pts[mid].bbox[0] + pts[mid].bbox[2]) / 2, (pts[mid].bbox[1] + pts[mid].bbox[3]) / 2);
          const [nx, ny] = toCanvas((pts[mid + 1].bbox[0] + pts[mid + 1].bbox[2]) / 2, (pts[mid + 1].bbox[1] + pts[mid + 1].bbox[3]) / 2);
          const angle = Math.atan2(ny - my, nx - mx);
          ctx.save();
          ctx.translate(mx, my);
          ctx.rotate(angle);
          ctx.fillStyle = color;
          ctx.globalAlpha = 0.8;
          ctx.beginPath();
          ctx.moveTo(8, 0);
          ctx.lineTo(-4, -5);
          ctx.lineTo(-4, 5);
          ctx.closePath();
          ctx.fill();
          ctx.restore();
          ctx.globalAlpha = 1.0;
        }

        if (showLabels) {
          ctx.font = "bold 10px monospace";
          ctx.fillStyle = color;
          ctx.fillText(`#${t.track_id} ${t.class_name.toUpperCase()} ${(t.source_model || "yolo11").toUpperCase()}`, startX + 8, startY - 5);
        }
      });
    }

    if (showIncidents && incidents) {
      incidents.forEach(inc => {
        const tracksInvolved = inc.involved_track_ids || [];
        let incX = W / 2, incY = H / 2;
        if (tracksInvolved.length > 0) {
          const targetTrack = tracks.find(t => t.track_id === tracksInvolved[0]);
          if (targetTrack && targetTrack.points) {
            const matchingPt = targetTrack.points.find(p => Math.abs(p.timestamp - inc.timestamp) < 0.5) || targetTrack.points[0];
            if (matchingPt) {
              [incX, incY] = toCanvas((matchingPt.bbox[0] + matchingPt.bbox[2]) / 2, (matchingPt.bbox[1] + matchingPt.bbox[3]) / 2);
            }
          }
        }
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.4;
        ctx.beginPath(); ctx.arc(incX, incY, 16, 0, Math.PI * 2); ctx.stroke();
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = "#ef4444";
        ctx.beginPath(); ctx.arc(incX, incY, 8, 0, Math.PI * 2); ctx.fill();
        ctx.font = "bold 10px sans-serif";
        ctx.fillText("INCIDENT", incX + 12, incY + 4);
      });
    }

    ctx.globalAlpha = 0.9;
    ctx.fillStyle = "rgba(17,24,39,0.9)";
    ctx.fillRect(8, H - 62, 180, 54);
    ctx.font = "9px monospace";
    ctx.fillStyle = "#10b981"; ctx.beginPath(); ctx.arc(18, H - 48, 4, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#94a3b8"; ctx.fillText("Entry Point", 28, H - 44);
    ctx.fillStyle = "#ef4444"; ctx.beginPath(); ctx.arc(18, H - 33, 4, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#94a3b8"; ctx.fillText("Exit Point", 28, H - 29);
    ctx.fillStyle = "#f59e0b"; ctx.beginPath(); ctx.arc(18, H - 18, 4, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#94a3b8"; ctx.fillText("Incident Location", 28, H - 14);
    ctx.globalAlpha = 1.0;

  }, [tracks, incidents, highlightedTrackId, showTrails, showLabels, showIncidents, getTrackColor]);

  return (
    <canvas 
      ref={canvasRef} 
      width={800} 
      height={500} 
      className="w-full h-full rounded-lg"
      style={{ background: "#05070a" }}
    />
  );
}

// ─── FullVideoMap ───────────────────────────────────────────────
function FullVideoMap({ data, onClose, initialSeekTime, initialHighlightTrackId }) {
  const [activeTab, setActiveTab] = useState("video");
  const [showLabels, setShowLabels] = useState(true);
  const [showTrails, setShowTrails] = useState(true);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showIncidents, setShowIncidents] = useState(true);
  
  const [currentTime, setCurrentTime] = useState(0);
  const [naturalWidth, setNaturalWidth] = useState(1920);
  const [naturalHeight, setNaturalHeight] = useState(1080);
  const [displayWidth, setDisplayWidth] = useState(0);
  const [displayHeight, setDisplayHeight] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  
  const [highlightedTrackId, setHighlightedTrackId] = useState(initialHighlightTrackId || null);
  const [hoveredTrackId, setHoveredTrackId] = useState(null);
  const [rawPage, setRawPage] = useState(0);
  const RAW_PAGE_SIZE = 100;
  
  const videoRef = useRef(null);
  const containerRef = useRef(null);
  const animFrameRef = useRef(null);

  // requestAnimationFrame sync
  useEffect(() => {
    const tick = () => {
      if (videoRef.current && !videoRef.current.paused) {
        setCurrentTime(videoRef.current.currentTime);
      }
      animFrameRef.current = requestAnimationFrame(tick);
    };
    animFrameRef.current = requestAnimationFrame(tick);
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, []);

  // Initial seek
  useEffect(() => {
    if (initialSeekTime != null && videoRef.current) {
      const trySeek = () => {
        if (videoRef.current && videoRef.current.readyState >= 1) {
          videoRef.current.currentTime = initialSeekTime;
          setCurrentTime(initialSeekTime);
        } else {
          setTimeout(trySeek, 200);
        }
      };
      trySeek();
    }
  }, [initialSeekTime]);

  // Resize listener
  useEffect(() => {
    const updateDims = () => {
      if (videoRef.current) {
        setDisplayWidth(videoRef.current.clientWidth);
        setDisplayHeight(videoRef.current.clientHeight);
      }
    };
    window.addEventListener("resize", updateDims);
    const interval = setInterval(updateDims, 300);
    return () => {
      window.removeEventListener("resize", updateDims);
      clearInterval(interval);
    };
  }, [activeTab]);

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setNaturalWidth(videoRef.current.videoWidth || 1920);
      setNaturalHeight(videoRef.current.videoHeight || 1080);
      setDisplayWidth(videoRef.current.clientWidth);
      setDisplayHeight(videoRef.current.clientHeight);
    }
  };

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play().catch(() => {});
      }
      setIsPlaying(!isPlaying);
    }
  };

  const seekTo = (t) => {
    if (videoRef.current) {
      videoRef.current.currentTime = t;
      setCurrentTime(t);
    }
  };

  const getTrackColor = useCallback((trackId) => {
    if (trackId === -1) return "#ef4444";
    const colors = ["#3b82f6", "#10b981", "#ec4899", "#8b5cf6", "#f59e0b", "#06b6d4", "#eab308", "#a855f7"];
    return colors[Math.abs(trackId) % colors.length];
  }, []);

  const isLegacy = !data.tracks || data.tracks.length === 0 || data.tracks.every(t => !t.points || t.points.length === 0);

  const getInterpolatedPoint = useCallback((points, time) => {
    if (!points || points.length === 0) return null;
    const sorted = [...points].sort((a, b) => a.timestamp - b.timestamp);
    let before = null, after = null;
    for (let i = 0; i < sorted.length; i++) {
      if (sorted[i].timestamp <= time) before = sorted[i];
      if (sorted[i].timestamp >= time && !after) after = sorted[i];
    }
    if (!before && !after) return null;
    if (!before) return after;
    if (!after) return before;
    if (before === after) return before;
    const dt = after.timestamp - before.timestamp;
    if (dt === 0) return before;
    const factor = (time - before.timestamp) / dt;
    return {
      bbox: [
        before.bbox[0] + (after.bbox[0] - before.bbox[0]) * factor,
        before.bbox[1] + (after.bbox[1] - before.bbox[1]) * factor,
        before.bbox[2] + (after.bbox[2] - before.bbox[2]) * factor,
        before.bbox[3] + (after.bbox[3] - before.bbox[3]) * factor,
      ],
      speed: before.speed + (after.speed - before.speed) * factor,
      confidence: before.confidence + (after.confidence - before.confidence) * factor
    };
  }, []);

  const activeOverlays = useMemo(() => {
    if (isLegacy) return [];
    const overlays = [];
    data.tracks.forEach(track => {
      if (currentTime >= track.first_seen && currentTime <= track.last_seen) {
        const pt = getInterpolatedPoint(track.points, currentTime);
        if (pt) overlays.push({ track, bbox: pt.bbox, speed: pt.speed, confidence: pt.confidence });
      }
    });
    return overlays;
  }, [isLegacy, data.tracks, currentTime, getInterpolatedPoint]);

  const scaleX = displayWidth / naturalWidth;
  const scaleY = displayHeight / naturalHeight;

  const renderHistoricalPath = useCallback((points) => {
    if (!points || points.length === 0) return "";
    const historical = points.filter(p => p.timestamp <= currentTime);
    return historical.map(p => {
      const cx = ((p.bbox[0] + p.bbox[2]) / 2) * scaleX;
      const cy = ((p.bbox[1] + p.bbox[3]) / 2) * scaleY;
      return `${cx},${cy}`;
    }).join(" ");
  }, [currentTime, scaleX, scaleY]);

  const incidents = useMemo(() => {
    return (data.events || []).filter(e => e.event_type && e.event_type.startsWith("possible_"));
  }, [data.events]);

  const trackStats = useMemo(() => {
    if (!data.tracks) return [];
    return data.tracks.map(t => {
      const pts = t.points || [];
      const speeds = pts.map(p => p.speed || 0);
      const avgSpeed = speeds.length > 0 ? speeds.reduce((a, b) => a + b, 0) / speeds.length : 0;
      const maxSpeed = speeds.length > 0 ? Math.max(...speeds) : 0;
      const avgConf = pts.length > 0 ? pts.reduce((s, p) => s + p.confidence, 0) / pts.length : 0;
      let distance = 0;
      for (let i = 1; i < pts.length; i++) {
        const dx = ((pts[i].bbox[0] + pts[i].bbox[2]) / 2) - ((pts[i-1].bbox[0] + pts[i-1].bbox[2]) / 2);
        const dy = ((pts[i].bbox[1] + pts[i].bbox[3]) / 2) - ((pts[i-1].bbox[1] + pts[i-1].bbox[3]) / 2);
        distance += Math.sqrt(dx * dx + dy * dy);
      }
      return { ...t, avgSpeed, maxSpeed, avgConf, distance, totalPoints: pts.length };
    });
  }, [data.tracks]);

  const allRawPoints = useMemo(() => {
    if (!data.tracks) return [];
    const all = [];
    data.tracks.forEach(t => {
      (t.points || []).forEach(p => { all.push({ track_id: t.track_id, class_name: t.class_name, ...p }); });
    });
    all.sort((a, b) => a.timestamp - b.timestamp);
    return all;
  }, [data.tracks]);

  const rawPageData = allRawPoints.slice(rawPage * RAW_PAGE_SIZE, (rawPage + 1) * RAW_PAGE_SIZE);
  const rawTotalPages = Math.ceil(allRawPoints.length / RAW_PAGE_SIZE);

  return (
    <div className="flex flex-col h-full bg-[#0a0e17] text-white overflow-hidden p-6 gap-6 relative">
      
      {/* Top Header */}
      <div className="flex justify-between items-center bg-[#111827]/80 border border-white/10 rounded-xl p-4 shrink-0 shadow-lg backdrop-blur-sm">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <span className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[10px] uppercase font-bold px-2 py-0.5 rounded">FULL VIDEO MAPPING MODE</span>
            <span className="text-xs text-slate-400 font-mono">{data.filename} &middot; {data.video_id.substring(0, 8)}</span>
          </div>
          <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            <Activity className="w-5 h-5 text-emerald-400 animate-pulse" />
            Visual Surveillance Reconstruction
          </h2>
        </div>
        <button onClick={onClose} className="bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg px-3 py-1.5 text-xs font-semibold flex items-center gap-1 transition-colors text-slate-300">
          <X className="w-4 h-4" /> Exit Mapping View
        </button>
      </div>

      {/* Meta Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 shrink-0">
        <div className="bg-[#111827]/50 border border-white/5 p-3.5 rounded-xl flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center border border-blue-500/20"><Clock className="w-5 h-5 text-blue-400" /></div>
          <div><div className="text-[10px] uppercase text-slate-400 font-bold tracking-wider">Duration</div><div className="text-sm font-bold font-mono">{data.duration?.toFixed(1)} sec</div></div>
        </div>
        <div className="bg-[#111827]/50 border border-white/5 p-3.5 rounded-xl flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-purple-500/10 flex items-center justify-center border border-purple-500/20"><Activity className="w-5 h-5 text-purple-400" /></div>
          <div><div className="text-[10px] uppercase text-slate-400 font-bold tracking-wider">Surveillance FPS</div><div className="text-sm font-bold font-mono">{data.fps ? Math.round(data.fps) : 24}</div></div>
        </div>
        <div className="bg-[#111827]/50 border border-white/5 p-3.5 rounded-xl flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-emerald-500/10 flex items-center justify-center border border-emerald-500/20"><Camera className="w-5 h-5 text-emerald-400" /></div>
          <div><div className="text-[10px] uppercase text-slate-400 font-bold tracking-wider">Total Frames</div><div className="text-sm font-bold font-mono">{data.frame_count || Math.round(data.duration * 24)}</div></div>
        </div>
        <div className="bg-[#111827]/50 border border-white/5 p-3.5 rounded-xl flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-amber-500/10 flex items-center justify-center border border-amber-500/20"><Sparkles className="w-5 h-5 text-amber-400" /></div>
          <div><div className="text-[10px] uppercase text-slate-400 font-bold tracking-wider">Unique Tracks</div><div className="text-sm font-bold font-mono">{data.tracks ? data.tracks.length : 0}</div></div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-white/10 shrink-0">
        {[
          { id: "video", label: "Annotated Video", icon: Video },
          { id: "tracks", label: "Objects & Tracks", icon: ChevronRight },
          { id: "timeline", label: "Timeline", icon: Clock },
          { id: "map", label: "Map / Trajectory View", icon: MapPin },
          { id: "raw", label: "Raw Data", icon: FileText }
        ].map(t => (
          <button key={t.id} onClick={() => setActiveTab(t.id)} className={`flex items-center gap-1.5 px-4 py-2 text-xs font-bold border-b-2 transition-all capitalize ${activeTab === t.id ? "border-emerald-500 text-emerald-400 bg-emerald-500/5" : "border-transparent text-slate-400 hover:text-white"}`}>
            <t.icon className="w-3.5 h-3.5" />{t.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="flex-1 min-h-0 relative">
        
        {isLegacy && (
          <div className="absolute top-4 left-4 right-4 bg-amber-500/10 border border-amber-500/20 rounded-xl p-4 z-20 flex gap-3 text-amber-400 text-xs">
            <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
            <div>
              <h4 className="font-bold mb-1">Dense tracking unavailable</h4>
              <p>This video was processed before track persistence was enabled. Re-analyze this video to enable the full visual mapping features.</p>
            </div>
          </div>
        )}

        {/* TAB 1: Annotated Video */}
        {activeTab === "video" && (
          <div className="w-full h-full flex flex-col lg:flex-row gap-6 relative">
            <div className="flex-1 flex flex-col bg-black rounded-xl border border-white/10 overflow-hidden relative shadow-2xl">
              <div ref={containerRef} className="relative flex-1 flex items-center justify-center overflow-hidden bg-[#05070a]">
                <video ref={videoRef} src={`http://localhost:8080/storage/videos/${data.video_id}/${data.filename}`} onLoadedMetadata={handleLoadedMetadata} className="w-full h-full object-contain" onClick={togglePlay} />
                
                {!isLegacy && showBoxes && displayWidth > 0 && displayHeight > 0 && (
                  <div className="absolute inset-0 pointer-events-none" style={{ width: displayWidth, height: displayHeight, margin: 'auto', left: 0, right: 0, top: 0, bottom: 0 }}>
                    
                    {showTrails && (
                      <svg className="absolute inset-0 w-full h-full">
                        {data.tracks.map(t => {
                          const pathStr = renderHistoricalPath(t.points);
                          if (!pathStr) return null;
                          const color = getTrackColor(t.track_id);
                          const isHL = highlightedTrackId === t.track_id;
                          return (<polyline key={t.track_id} points={pathStr} fill="none" stroke={color} strokeWidth={isHL ? 3 : 1.5} strokeDasharray={isHL ? "none" : "3,3"} className="transition-all duration-100" />);
                        })}
                      </svg>
                    )}
                    
                    {activeOverlays.map(({ track, bbox, speed, confidence }) => {
                      const color = getTrackColor(track.track_id);
                      const isHL = highlightedTrackId === track.track_id;
                      const speedLabel = track.stationary ? "Stationary" : (speed > 0 ? `${speed.toFixed(1)} px/s` : "Unknown");
                      const left = bbox[0] * scaleX;
                      const top = bbox[1] * scaleY;
                      const width = (bbox[2] - bbox[0]) * scaleX;
                      const height = (bbox[3] - bbox[1]) * scaleY;
                      
                      return (
                        <div key={track.track_id} style={{ position: 'absolute', left, top, width, height, border: `2px solid ${color}`, boxShadow: isHL ? `0 0 12px ${color}` : 'none' }} className="transition-all duration-75 flex flex-col justify-start pointer-events-auto cursor-pointer" onMouseEnter={() => setHoveredTrackId(track.track_id)} onMouseLeave={() => setHoveredTrackId(null)} onClick={() => setHighlightedTrackId(track.track_id)}>
                          {showLabels && (
                            <div style={{ backgroundColor: color }} className="absolute top-0 left-0 -translate-y-full text-[9px] font-bold text-white px-1.5 py-0.5 rounded-t whitespace-nowrap shadow flex flex-col gap-0.5">
                              <span>#{track.track_id} {track.class_name.toUpperCase()}</span>
                              {track.color && track.color !== "Unknown" && <span>{track.color}</span>}
                              {track.brand && track.brand !== "Unknown" && <span>{track.brand}</span>}
                              {track.model && track.model !== "Unknown" && <span>{track.model}</span>}
                              <span>{speedLabel}</span>
                              <span>Source: {(track.source_model || "yolo11").toUpperCase()}</span>
                              <span>Conf: {(confidence * 100).toFixed(0)}%</span>
                            </div>
                          )}
                        </div>
                      );
                    })}

                    {incidents.map(inc => {
                      const isTimeClose = Math.abs(currentTime - inc.timestamp) < 1.0;
                      if (!isTimeClose) return null;
                      const reason = inc.reason || {};
                      const status = reason.status || "POSSIBLE";
                      return (
                        <div key={inc.event_id} className={`absolute top-12 left-1/2 -translate-x-1/2 font-bold text-xs uppercase px-3 py-1.5 rounded-lg border shadow-lg animate-bounce ${status === "CONFIRMED" ? "bg-red-600/90 text-white border-red-500 shadow-[0_0_15px_rgba(239,68,68,0.5)]" : "bg-amber-600/90 text-white border-amber-500 shadow-[0_0_15px_rgba(245,158,11,0.5)]"}`}>
                          {status} {inc.event_type.replace('possible_', '').replace('_', ' ')}: {inc.timestamp.toFixed(2)}s
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="bg-[#0f141e] border-t border-white/10 p-3 flex items-center justify-between shrink-0 font-mono text-xs">
                <div className="flex items-center gap-3">
                  <button onClick={togglePlay} className="bg-emerald-600 hover:bg-emerald-500 rounded px-3 py-1 text-white font-semibold flex items-center gap-1 transition-colors">{isPlaying ? "⏸ Pause" : "▶ Play"}</button>
                  <span className="text-slate-400">Time: {currentTime.toFixed(2)}s / {data.duration?.toFixed(2)}s</span>
                </div>
                {!isLegacy && (
                  <div className="flex items-center gap-4 text-[10px] text-slate-400">
                    <label className="flex items-center gap-1.5 cursor-pointer"><input type="checkbox" checked={showBoxes} onChange={e => setShowBoxes(e.target.checked)} className="rounded bg-slate-800 border-white/10 text-emerald-500 focus:ring-0 w-3 h-3" /><span>Boxes</span></label>
                    <label className="flex items-center gap-1.5 cursor-pointer"><input type="checkbox" checked={showLabels} onChange={e => setShowLabels(e.target.checked)} className="rounded bg-slate-800 border-white/10 text-emerald-500 focus:ring-0 w-3 h-3" /><span>Labels</span></label>
                    <label className="flex items-center gap-1.5 cursor-pointer"><input type="checkbox" checked={showTrails} onChange={e => setShowTrails(e.target.checked)} className="rounded bg-slate-800 border-white/10 text-emerald-500 focus:ring-0 w-3 h-3" /><span>Trails</span></label>
                  </div>
                )}
              </div>
            </div>

            {/* Right: Incident Logs */}
            <div className="w-full lg:w-80 flex flex-col bg-[#111827]/60 border border-white/10 rounded-xl overflow-hidden shadow-xl">
              <div className="p-3 bg-[#111827] border-b border-white/10 flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-red-400 animate-pulse" />
                <h3 className="font-bold text-xs uppercase tracking-wider">Safety &amp; Incident Logs</h3>
              </div>
              <div className="flex-1 overflow-y-auto p-3 space-y-3">
                {incidents.length === 0 ? (
                  <div className="text-center text-xs text-slate-500 py-8">No safety incidents flagged.</div>
                ) : (
                  incidents.map(inc => {
                    const reason = inc.reason || {};
                    const evidence = reason.evidence || {};
                    const status = reason.status || "POSSIBLE";
                    return (
                      <div key={inc.event_id} onClick={() => seekTo(inc.timestamp)} className={`p-3 rounded-lg border cursor-pointer hover:bg-white/5 transition-all flex flex-col gap-2 ${status === "CONFIRMED" ? "border-red-500/30 bg-red-500/5" : "border-amber-500/30 bg-amber-500/5"}`}>
                        <div className="flex justify-between items-start">
                          <span className={`text-[10px] font-bold uppercase px-1.5 py-0.5 rounded ${status === "CONFIRMED" ? "bg-red-500/10 text-red-400 border border-red-500/20" : "bg-amber-500/10 text-amber-400 border border-amber-500/20"}`}>{status} INCIDENT</span>
                          <span className="font-mono text-xs font-bold text-slate-400">{inc.timestamp.toFixed(2)}s</span>
                        </div>
                        <div className="text-xs font-bold capitalize text-white">{inc.event_type.replace("possible_", "").replace("_", " ")}</div>
                        <div className="text-[10px] text-slate-400 leading-normal">{inc.description}</div>
                        <div className="border-t border-white/5 pt-2 mt-1 space-y-1 text-[9px] text-slate-400 font-mono">
                          <div className="flex justify-between"><span>Geometry:</span><span className="text-slate-300 font-bold">{reason.geometry_score || "0.0"}</span></div>
                          <div className="flex justify-between"><span>Motion:</span><span className="text-slate-300 font-bold">{reason.motion_score || "0.0"}</span></div>
                          <div className="flex justify-between"><span>Temporal:</span><span className="text-slate-300 font-bold">{reason.temporal_score || "0.0"}</span></div>
                          {evidence.minimum_distance !== undefined && <div className="flex justify-between"><span>Min Dist:</span><span className="text-slate-300">{evidence.minimum_distance}px</span></div>}
                          {evidence.iou !== undefined && <div className="flex justify-between"><span>IoU:</span><span className="text-slate-300">{evidence.iou}</span></div>}
                          <div className="flex justify-between"><span>Deceleration:</span><span className={evidence.deceleration ? "text-green-400 font-bold" : "text-slate-500"}>{evidence.deceleration ? "Yes" : "No"}</span></div>
                          {evidence.videomae_action && <div className="flex justify-between"><span>VideoMAE:</span><span className="text-slate-300">{evidence.videomae_action} ({((evidence.videomae_confidence || 0) * 100).toFixed(0)}%)</span></div>}
                          <div className="flex justify-between pt-1 border-t border-white/5 text-[10px]"><span className="font-bold text-slate-300">Confidence:</span><span className="text-emerald-400 font-bold">{((reason.final_confidence || inc.confidence) * 100).toFixed(1)}%</span></div>
                          {reason.involved_tracks && <div className="flex justify-between text-[10px]"><span>Involved:</span><span className="text-blue-400 font-bold">{reason.involved_tracks.map(id => `#${id}`).join(", ")}</span></div>}
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: Objects & Tracks */}
        {activeTab === "tracks" && (
          <div className="w-full h-full bg-[#111827]/40 border border-white/10 rounded-xl overflow-hidden flex flex-col shadow-xl">
            <div className="overflow-x-auto overflow-y-auto flex-1">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-[#111827] text-slate-400 uppercase font-bold text-[10px] tracking-wider border-b border-white/10 sticky top-0 z-10">
                    <th className="p-4">Track ID</th><th className="p-4">Class</th><th className="p-4">Source</th><th className="p-4">Color</th><th className="p-4">Brand / Model</th><th className="p-4">First Seen</th><th className="p-4">Last Seen</th><th className="p-4">Avg Speed</th><th className="p-4">Max Speed</th><th className="p-4">Status</th><th className="p-4">Points</th><th className="p-4">Confidence</th><th className="p-4">Distance</th><th className="p-4">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 text-slate-300">
                  {trackStats.length === 0 ? (
                    <tr><td colSpan="14" className="p-8 text-center text-slate-500">No tracking data.</td></tr>
                  ) : (
                    trackStats.map(t => (
                      <tr key={t.track_id} className={`hover:bg-white/5 transition-colors cursor-pointer ${highlightedTrackId === t.track_id ? "bg-emerald-500/5 text-emerald-300 font-semibold" : ""}`} onClick={() => { setHighlightedTrackId(t.track_id); seekTo(t.first_seen); setActiveTab("video"); }}>
                        <td className="p-4 font-mono font-bold"><span className="inline-block w-3 h-3 rounded-full mr-2" style={{ backgroundColor: getTrackColor(t.track_id) }}></span>#{t.track_id}</td>
                        <td className="p-4 capitalize font-semibold">{t.class_name}</td>
                        <td className="p-4 uppercase text-[10px] font-bold">{t.source_model || "yolo11"}</td>
                        <td className="p-4 capitalize">{t.color || "?"}</td>
                        <td className="p-4 capitalize">{t.brand || "?"}{t.model && t.model !== "Unknown" ? ` / ${t.model}` : ""}</td>
                        <td className="p-4 font-mono">{t.first_seen?.toFixed(2)}s</td>
                        <td className="p-4 font-mono">{t.last_seen?.toFixed(2)}s</td>
                        <td className="p-4 font-mono">{t.avgSpeed.toFixed(1)}</td>
                        <td className="p-4 font-mono">{t.maxSpeed.toFixed(1)}</td>
                        <td className="p-4 font-mono">{t.stationary ? "Stationary" : "Moving/Unknown"}</td>
                        <td className="p-4 font-mono">{t.totalPoints}</td>
                        <td className="p-4 font-mono">{(t.avgConf * 100).toFixed(0)}%</td>
                        <td className="p-4 font-mono">{t.distance.toFixed(0)}px</td>
                        <td className="p-4"><button onClick={(e) => { e.stopPropagation(); setHighlightedTrackId(t.track_id); seekTo(t.first_seen); setActiveTab("video"); }} className="bg-emerald-600/10 hover:bg-emerald-600/20 text-emerald-400 border border-emerald-500/20 rounded px-2.5 py-1 font-semibold text-[10px]">Jump</button></td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* TAB 3: Timeline */}
        {activeTab === "timeline" && (
          <div className="w-full h-full bg-[#111827]/40 border border-white/10 rounded-xl p-6 overflow-y-auto shadow-xl space-y-4">
            {(() => {
              const chronology = [];
              if (data.tracks) {
                data.tracks.forEach(t => {
                  chronology.push({ time: t.first_seen, type: "entry", title: `${t.class_name.toUpperCase()} Entered`, description: `Track #${t.track_id} (${t.color || '?'}, ${t.brand || '?'}) first observed via ${(t.source_model || 'yolo11').toUpperCase()}.`, trackId: t.track_id });
                  chronology.push({ time: t.last_seen, type: "exit", title: `${t.class_name.toUpperCase()} Exited`, description: `Track #${t.track_id} left view.`, trackId: t.track_id });
                });
              }
              incidents.forEach(inc => {
                const reason = inc.reason || {};
                const status = reason.status || "POSSIBLE";
                chronology.push({ time: inc.timestamp, type: status === "CONFIRMED" ? "confirmed_incident" : "possible_incident", title: `${inc.event_type.replace('possible_', '').replace('_', ' ').toUpperCase()}`, description: inc.description, trackId: -1, confidence: inc.confidence, reason });
              });
              chronology.sort((a, b) => a.time - b.time);
              if (chronology.length === 0) return <div className="text-center text-slate-500 py-12">No events.</div>;
              return (
                <div className="relative border-l border-white/10 pl-6 space-y-6">
                  {chronology.map((evt, idx) => {
                    const isInc = evt.type.includes("incident");
                    const isConf = evt.type === "confirmed_incident";
                    return (
                      <div key={idx} onClick={() => { seekTo(evt.time); setActiveTab("video"); if (evt.trackId > 0) setHighlightedTrackId(evt.trackId); }} className="relative cursor-pointer group hover:opacity-90">
                        <div className={`absolute top-1 -left-[31px] w-4 h-4 rounded-full border-2 bg-[#0a0e17] flex items-center justify-center transition-all ${isInc ? (isConf ? "border-red-500 shadow-[0_0_5px_#ef4444]" : "border-amber-500 shadow-[0_0_5px_#f59e0b]") : "border-emerald-500 group-hover:bg-emerald-500/20"}`}>
                          {isInc && <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-ping"></span>}
                        </div>
                        <div className="bg-[#111827]/50 border border-white/5 rounded-lg p-4 group-hover:border-white/10 transition-colors">
                          <div className="flex justify-between items-start mb-1">
                            <h4 className={`text-xs font-bold tracking-tight uppercase ${isInc ? (isConf ? "text-red-400" : "text-amber-400") : "text-emerald-400"}`}>{evt.title}</h4>
                            <span className="font-mono text-xs font-bold text-slate-400">{evt.time?.toFixed(2)}s</span>
                          </div>
                          <p className="text-xs text-slate-300 leading-relaxed mb-2">{evt.description}</p>
                          {isInc && evt.reason && typeof evt.reason === "object" && (
                            <div className="text-[10px] text-slate-500 font-mono flex flex-wrap gap-x-3 gap-y-1 mt-1 border-t border-white/5 pt-2">
                              <span>Geom: {evt.reason.geometry_score}</span>
                              <span>Motion: {evt.reason.motion_score}</span>
                              <span>Temporal: {evt.reason.temporal_score}</span>
                              <span>Conf: {((evt.reason.final_confidence || 0) * 100).toFixed(1)}%</span>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              );
            })()}
          </div>
        )}

        {/* TAB 4: Map View (Canvas) */}
        {activeTab === "map" && (
          <div className="w-full h-full flex flex-col md:flex-row gap-6 relative">
            <div className="flex-1 bg-black/60 border border-white/10 rounded-xl relative overflow-hidden flex items-center justify-center shadow-2xl">
              {isLegacy ? (
                <div className="text-slate-500 text-center text-xs">Trajectories unavailable.</div>
              ) : (
                <MapCanvas tracks={data.tracks} incidents={incidents} highlightedTrackId={highlightedTrackId} showTrails={showTrails} showLabels={showLabels} showIncidents={showIncidents} getTrackColor={getTrackColor} />
              )}
            </div>
            <div className="w-full md:w-64 bg-[#111827]/60 border border-white/10 rounded-xl p-4 flex flex-col gap-4 shadow-xl">
               <h4 className="font-bold text-xs uppercase tracking-wider border-b border-white/10 pb-2">Map Controls</h4>
               <div className="space-y-3 text-xs">
                 <label className="flex items-center gap-2 cursor-pointer text-slate-300"><input type="checkbox" checked={showTrails} onChange={e => setShowTrails(e.target.checked)} className="rounded bg-slate-800 border-white/10 text-emerald-500 focus:ring-0 w-3.5 h-3.5" />Show/Hide Trails</label>
                 <label className="flex items-center gap-2 cursor-pointer text-slate-300"><input type="checkbox" checked={showLabels} onChange={e => setShowLabels(e.target.checked)} className="rounded bg-slate-800 border-white/10 text-emerald-500 focus:ring-0 w-3.5 h-3.5" />Show/Hide Labels</label>
                 <label className="flex items-center gap-2 cursor-pointer text-slate-300"><input type="checkbox" checked={showIncidents} onChange={e => setShowIncidents(e.target.checked)} className="rounded bg-slate-800 border-white/10 text-emerald-500 focus:ring-0 w-3.5 h-3.5" />Show/Hide Incidents</label>
               </div>
               <div className="border-t border-white/10 pt-3 mt-2">
                 <h4 className="font-bold text-xs uppercase tracking-wider mb-2">Track Legend</h4>
                 <div className="space-y-1.5 max-h-64 overflow-y-auto">
                   {(data.tracks || []).map(t => (
                     <div key={t.track_id} className={`flex items-center gap-2 text-[10px] p-1.5 rounded cursor-pointer hover:bg-white/5 ${highlightedTrackId === t.track_id ? "bg-white/10" : ""}`} onClick={() => { setHighlightedTrackId(t.track_id); seekTo(t.first_seen); setActiveTab("video"); }}>
                       <span className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: getTrackColor(t.track_id) }}></span>
                       <span className="font-mono font-bold">#{t.track_id}</span>
                       <span className="text-slate-400 capitalize truncate">{t.class_name} &middot; {t.color || "?"}</span>
                     </div>
                   ))}
                 </div>
               </div>
               <p className="text-[10px] text-slate-500 leading-relaxed mt-auto border-t border-white/10 pt-3">Green dots = entry, red dots = exit. Click any track to jump to its video frame.</p>
            </div>
          </div>
        )}

        {/* TAB 5: Raw Data (Paginated) */}
        {activeTab === "raw" && (
          <div className="w-full h-full bg-[#111827]/40 border border-white/10 rounded-xl overflow-hidden flex flex-col shadow-xl">
            <div className="p-3 bg-[#111827] border-b border-white/10 flex items-center justify-between">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Raw Track Points &mdash; {allRawPoints.length} total</span>
              <div className="flex items-center gap-2 text-xs">
                <button onClick={() => setRawPage(Math.max(0, rawPage - 1))} disabled={rawPage === 0} className="px-2 py-1 bg-white/5 hover:bg-white/10 disabled:opacity-30 rounded border border-white/10">&larr; Prev</button>
                <span className="text-slate-400 font-mono">Page {rawPage + 1} / {rawTotalPages || 1}</span>
                <button onClick={() => setRawPage(Math.min(rawTotalPages - 1, rawPage + 1))} disabled={rawPage >= rawTotalPages - 1} className="px-2 py-1 bg-white/5 hover:bg-white/10 disabled:opacity-30 rounded border border-white/10">Next &rarr;</button>
              </div>
            </div>
            <div className="overflow-auto flex-1">
              <table className="w-full text-left border-collapse text-[11px] font-mono">
                <thead>
                  <tr className="bg-[#111827] text-slate-500 uppercase text-[9px] tracking-wider border-b border-white/10 sticky top-0 z-10">
                    <th className="p-2.5">Track</th><th className="p-2.5">Class</th><th className="p-2.5">Source</th><th className="p-2.5">Frame</th><th className="p-2.5">Time</th><th className="p-2.5">BBox</th><th className="p-2.5">Speed</th><th className="p-2.5">Conf</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 text-slate-300">
                  {rawPageData.map((p, idx) => (
                    <tr key={`${p.track_id}_${p.frame}_${idx}`} className="hover:bg-white/5 cursor-pointer" onClick={() => { seekTo(p.timestamp); setHighlightedTrackId(p.track_id); setActiveTab("video"); }}>
                      <td className="p-2.5 font-bold">#{p.track_id}</td>
                      <td className="p-2.5 capitalize">{p.class_name}</td>
                      <td className="p-2.5 uppercase">{p.source_model || "yolo11"}</td>
                      <td className="p-2.5">{p.frame}</td>
                      <td className="p-2.5">{p.timestamp?.toFixed(3)}s</td>
                      <td className="p-2.5 text-[10px]">[{p.bbox?.map(v => Math.round(v)).join(", ")}]</td>
                      <td className="p-2.5">{(p.speed || 0).toFixed(1)}</td>
                      <td className="p-2.5">{((p.confidence || 0) * 100).toFixed(0)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}

// ─── VideoCard ───────────────────────────────────────────────────
function VideoCard({ video, onOpen, onMap, onArchive, onDelete }) {
  const videoId = String(video.video_id || "");
  const filename = String(video.filename || "Unnamed video");
  const duration = Number(video.duration || 0);

  return (
    <div className="bg-[#151b2b] border border-white/10 hover:border-blue-500/30 rounded-xl overflow-hidden shadow-lg transition-all flex flex-col group">
      <div className="relative w-full aspect-video bg-black flex items-center justify-center border-b border-white/5">
        <img src={`http://localhost:8080/videos/${videoId}/thumbnail`} className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" onError={(e) => { e.target.style.display = 'none'; }} />
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none group-hover:bg-black/20 transition-all">
           <Play className="w-12 h-12 text-white/50 group-hover:text-white/90 drop-shadow-lg" />
        </div>
        <div className="absolute bottom-2 right-2 bg-black/70 backdrop-blur text-white text-[10px] font-mono font-bold px-2 py-1 rounded">
          {duration.toFixed(1)}s
        </div>
        {video.is_archived && (
          <div className="absolute top-2 right-2 bg-amber-500/80 backdrop-blur text-white text-[10px] font-bold px-2 py-1 rounded uppercase">Archived</div>
        )}
      </div>
      <div className="p-4 flex flex-col gap-3 flex-1">
        <div>
          <h3 className="font-bold text-sm text-white truncate" title={filename}>{filename}</h3>
          <p className="text-[10px] text-slate-400 font-mono mt-0.5">{videoId.substring(0, 8)}</p>
        </div>
        
        <div className="grid grid-cols-2 gap-2 text-xs text-slate-300 mt-auto bg-black/20 rounded-lg p-2 border border-white/5">
           <div className="flex justify-between"><span>Objects:</span> <span className="font-mono font-bold">{video.stats?.objects || 0}</span></div>
           <div className="flex justify-between"><span>Tracks:</span> <span className="font-mono font-bold">{video.stats?.tracks || 0}</span></div>
           <div className="flex justify-between"><span>Events:</span> <span className="font-mono font-bold">{video.stats?.events || 0}</span></div>
           <div className="flex justify-between text-red-400"><span>Incidents:</span> <span className="font-mono font-bold">{video.stats?.incidents || 0}</span></div>
        </div>
        
        <div className="flex gap-2 mt-2 pt-3 border-t border-white/5">
           <button onClick={() => onOpen(videoId)} className="flex-1 bg-blue-600 hover:bg-blue-500 text-white font-bold py-1.5 rounded-md text-xs transition-colors">OPEN</button>
           <button onClick={() => onMap(videoId)} className="flex-1 bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-1.5 rounded-md text-xs transition-colors">MAP</button>
           <button onClick={() => onArchive(videoId, !video.is_archived)} className="flex-1 bg-white/5 hover:bg-white/10 text-slate-300 border border-white/10 font-bold py-1.5 rounded-md text-xs transition-colors">{video.is_archived ? "UNARCHIVE" : "ARCHIVE"}</button>
           <button onClick={() => onDelete(videoId)} className="px-3 bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/20 font-bold py-1.5 rounded-md text-xs transition-colors"><X className="w-4 h-4" /></button>
        </div>
      </div>
    </div>
  );
}

// ─── GalleryView ─────────────────────────────────────────────────
function GalleryView({ videos, onOpenVideo, onMapVideo, fetchVideos, onUpload, videosLoading, videosError }) {
  const [activeTab, setActiveTab] = useState("active");
  const [search, setSearch] = useState("");
  const [sortParam, setSortParam] = useState("newest");
  const [filterParam, setFilterParam] = useState("all");
  const [showDeleteModal, setShowDeleteModal] = useState(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const handleArchive = async (videoId, isArchived) => {
    try {
      await axios.post(`http://localhost:8080/videos/${videoId}/archive`, { is_archived: isArchived });
      fetchVideos();
    } catch (err) { console.error("Archive error", err); }
  };

  const handleDelete = async () => {
    if (!showDeleteModal) return;
    setIsDeleting(true);
    try {
      await axios.delete(`http://localhost:8080/videos/${showDeleteModal}`);
      fetchVideos();
    } catch (err) { console.error("Delete error", err); }
    finally { setIsDeleting(false); setShowDeleteModal(null); }
  };

  let filtered = videos.filter(v => {
    const isArchived = v.is_archived === true || v.is_archived === 1 || v.is_archived === "1";
    if (activeTab === "active" && isArchived) return false;
    if (activeTab === "archived" && !isArchived) return false;
    
    const filename = String(v.filename || "").toLowerCase();
    const videoId = String(v.video_id || "").toLowerCase();
    if (search && !filename.includes(search.toLowerCase()) && !videoId.includes(search.toLowerCase())) return false;
    
    if (filterParam === "incidents" && (v.stats?.incidents || 0) === 0) return false;
    if (filterParam === "no_incidents" && (v.stats?.incidents || 0) > 0) return false;
    
    return true;
  });

  filtered.sort((a, b) => {
    const aId = String(a.video_id || "");
    const bId = String(b.video_id || "");
    const aDuration = Number(a.duration || 0);
    const bDuration = Number(b.duration || 0);
    if (sortParam === "newest") return bId.localeCompare(aId);
    if (sortParam === "oldest") return aId.localeCompare(bId);
    if (sortParam === "longest") return bDuration - aDuration;
    if (sortParam === "shortest") return aDuration - bDuration;
    if (sortParam === "incidents") return (b.stats?.incidents || 0) - (a.stats?.incidents || 0);
    return 0;
  });

  return (
    <div className="flex-1 flex flex-col h-full bg-[#0a0e17] overflow-hidden">
      <div className="h-16 border-b border-white/10 flex items-center px-6 bg-[#0a0e17]/80 backdrop-blur-md sticky top-0 z-10 shrink-0 gap-6">
        <h1 className="font-bold text-xl flex items-center gap-2"><Camera className="w-5 h-5 text-blue-500" />CCTV Gallery</h1>
        
        <div className="flex gap-4 border-l border-white/10 pl-6 h-full items-center text-sm font-semibold">
           <button onClick={() => setActiveTab("active")} className={`h-full border-b-2 px-2 transition-colors ${activeTab === 'active' ? 'border-blue-500 text-blue-400' : 'border-transparent text-slate-400 hover:text-white'}`}>ACTIVE VIDEOS</button>
           <button onClick={() => setActiveTab("archived")} className={`h-full border-b-2 px-2 transition-colors ${activeTab === 'archived' ? 'border-amber-500 text-amber-400' : 'border-transparent text-slate-400 hover:text-white'}`}>ARCHIVED VIDEOS</button>
        </div>
        <button onClick={onUpload} className="ml-auto flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold py-2 px-3 rounded-lg transition-colors">
          <UploadCloud className="w-4 h-4" /> UPLOAD CCTV
        </button>
      </div>
      
      <div className="p-6 border-b border-white/5 bg-[#0f141e] flex flex-wrap items-center justify-between gap-4 shrink-0">
         <div className="relative w-full max-w-sm">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
            <input type="text" value={search} onChange={e => setSearch(e.target.value)} placeholder="Search by filename or ID..." className="w-full bg-black/40 border border-white/10 rounded-lg py-2 pl-9 pr-4 text-sm text-white focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500" />
         </div>
         
         <div className="flex items-center gap-3">
            <select value={filterParam} onChange={e => setFilterParam(e.target.value)} className="bg-black/40 border border-white/10 rounded-lg p-2 text-sm text-white focus:border-blue-500 focus:outline-none appearance-none px-4">
              <option value="all">All Videos</option>
              <option value="incidents">Has Incidents</option>
              <option value="no_incidents">No Incidents</option>
            </select>
            <select value={sortParam} onChange={e => setSortParam(e.target.value)} className="bg-black/40 border border-white/10 rounded-lg p-2 text-sm text-white focus:border-blue-500 focus:outline-none appearance-none px-4">
              <option value="newest">Newest First</option>
              <option value="oldest">Oldest First</option>
              <option value="longest">Longest</option>
              <option value="shortest">Shortest</option>
              <option value="incidents">Most Incidents</option>
            </select>
         </div>
      </div>

      <div className="flex-1 p-6 overflow-y-auto">
         {videosLoading ? (
            <div className="w-full h-full flex flex-col items-center justify-center text-slate-500 gap-3">
               <Loader2 className="w-8 h-8 animate-spin text-blue-400" />
               <p>Loading stored videos...</p>
            </div>
         ) : videosError && videos.length === 0 ? (
            <div className="w-full h-full flex flex-col items-center justify-center text-slate-500 gap-3 text-center">
               <AlertCircle className="w-10 h-10 text-red-400" />
               <p className="text-red-300">{videosError}</p>
               <button onClick={() => fetchVideos()} className="bg-blue-600 hover:bg-blue-500 text-white font-bold py-2 px-4 rounded-lg text-sm transition-colors">RETRY</button>
            </div>
         ) : filtered.length === 0 ? (
            <div className="w-full h-full flex flex-col items-center justify-center text-slate-500">
               <Video className="w-12 h-12 mb-3 text-slate-700" />
               <p>{videos.length === 0 ? "No stored videos found." : `No videos found in ${activeTab} view.`}</p>
               <button onClick={onUpload} className="mt-4 bg-blue-600 hover:bg-blue-500 text-white font-bold py-2 px-4 rounded-lg text-sm transition-colors">UPLOAD CCTV</button>
            </div>
         ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {filtered.map(v => (
                <VideoCard key={v.video_id} video={v} onOpen={onOpenVideo} onMap={onMapVideo} onArchive={handleArchive} onDelete={(id) => setShowDeleteModal(id)} />
              ))}
            </div>
         )}
      </div>

      {showDeleteModal && (
         <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
           <div className="bg-[#151b2b] border border-red-500/30 rounded-2xl shadow-[0_0_50px_rgba(239,68,68,0.2)] w-full max-w-md overflow-hidden relative p-6">
             <div className="flex flex-col items-center text-center gap-4">
                <div className="w-16 h-16 bg-red-500/10 rounded-full flex items-center justify-center">
                   <AlertTriangle className="w-8 h-8 text-red-500" />
                </div>
                <h3 className="font-bold text-xl text-white">Delete Video Data?</h3>
                <p className="text-sm text-slate-300">
                  Delete this CCTV video and all associated analysis? 
                  This will permanently remove the original video, track points, events, evidence clips, and FAISS vector embeddings.
                </p>
                <div className="text-[10px] font-mono text-slate-500 border border-white/10 rounded p-2 bg-black/40 w-full mt-2">
                   VIDEO ID: {showDeleteModal}
                </div>
                <div className="flex gap-3 w-full mt-4">
                   <button onClick={() => setShowDeleteModal(null)} className="flex-1 py-3 bg-white/5 hover:bg-white/10 rounded-lg text-white font-bold transition-colors">Cancel</button>
                   <button onClick={handleDelete} disabled={isDeleting} className="flex-1 py-3 bg-red-600 hover:bg-red-500 disabled:bg-slate-800 disabled:text-slate-500 rounded-lg text-white font-bold flex items-center justify-center gap-2 transition-colors">
                      {isDeleting ? <><Loader2 className="w-5 h-5 animate-spin" /> Deleting...</> : 'Delete Permanently'}
                   </button>
                </div>
             </div>
           </div>
         </div>
      )}
    </div>
  );
}

// ─── Home (Main Page Component) ──────────────────────────────────
export default function Home() {
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [videoFile, setVideoFile] = useState(null);
  const [cameraId, setCameraId] = useState("CAM_01");
  const [location, setLocation] = useState("Main Gate");
  const [uploadError, setUploadError] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [processingStatus, setProcessingStatus] = useState(null); 
  const [activeVideoId, setActiveVideoId] = useState(null);
  const [availableVideos, setAvailableVideos] = useState([]);
  const [videosLoading, setVideosLoading] = useState(true);
  const [videosError, setVideosError] = useState("");
  const [selectedVideoId, setSelectedVideoId] = useState("");
  const [activeView, setActiveView] = useState("gallery"); // "gallery" | "analysis"
  const [extendedObjectDetection, setExtendedObjectDetection] = useState(false);
  const [extendedScanStatus, setExtendedScanStatus] = useState(null);

  // Full Video Map state
  const [fullMapData, setFullMapData] = useState(null);
  const [fullMapSeekTime, setFullMapSeekTime] = useState(null);
  const [fullMapHighlightTrack, setFullMapHighlightTrack] = useState(null);

  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [chatSessions, setChatSessions] = useState([]);
  const [sidebarTab, setSidebarTab] = useState("timeline"); // "timeline" | "history"

  const [messages, setMessages] = useState([
    { role: "assistant", content: "Welcome. I can search your indexed CCTV footage.\n\nYou can ask:\n\"Find the red car\"\n\"Show the person wearing a red shirt\"\n\"What happened at 20 seconds?\"\n\"When did the red car pass?\"\n\"Show me fully mapped\"\n\"Show me the accident\"" }
  ]);
  const [chatInput, setChatInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);
  
  const scrollToBottom = () => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); };
  useEffect(() => { scrollToBottom(); }, [messages, isTyping]);

  const loadChatSessions = useCallback(async () => {
    try {
      const res = await axios.get(`http://localhost:8080/chat/sessions?video_id=${selectedVideoId || "ALL"}`);
      setChatSessions(res.data);
    } catch (err) {
      console.error("Failed to load chat sessions:", err);
    }
  }, [selectedVideoId]);

  useEffect(() => {
    loadChatSessions();
  }, [loadChatSessions]);

  const loadSessionMessages = async (sessionId) => {
    try {
      const res = await axios.get(`http://localhost:8080/chat/sessions/${sessionId}/messages`);
      const formatted = res.data.map(m => ({
        role: m.role,
        content: m.content,
        results: m.results
      }));
      setMessages(formatted.length > 0 ? formatted : [
        { role: "assistant", content: "Welcome. I can search your indexed CCTV footage.\n\nYou can ask:\n\"Find the red car\"\n\"Show the person wearing a red shirt\"\n\"What happened at 20 seconds?\"\n\"When did the red car pass?\"\n\"Show me fully mapped\"\n\"Show me the accident\"" }
      ]);
      setCurrentSessionId(sessionId);
    } catch (err) {
      console.error("Failed to load session messages:", err);
    }
  };

  const handleNewChat = () => {
    setCurrentSessionId(null);
    setMessages([
      { role: "assistant", content: "Welcome. I can search your indexed CCTV footage.\n\nYou can ask:\n\"Find the red car\"\n\"Show the person wearing a red shirt\"\n\"What happened at 20 seconds?\"\n\"When did the red car pass?\"\n\"Show me fully mapped\"\n\"Show me the accident\"" }
    ]);
  };

  const handleDeleteSession = async (e, sessionId) => {
    e.stopPropagation();
    if (!window.confirm("Are you sure you want to delete this investigation log?")) return;
    try {
      await axios.delete(`http://localhost:8080/chat/sessions/${sessionId}`);
      if (currentSessionId === sessionId) {
        handleNewChat();
      }
      loadChatSessions();
    } catch (err) {
      console.error("Failed to delete chat session:", err);
    }
  };

  const fetchVideos = useCallback(async ({ silent = false } = {}) => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    if (!silent) {
      setVideosLoading(true);
      setVideosError("");
    }
    try {
      const res = await axios.get("http://localhost:8080/videos", { signal: controller.signal });
      const videos = Array.isArray(res.data) ? res.data : [];
      setAvailableVideos(videos);
      setVideosError("");
      setSelectedVideoId(current => current || videos[0]?.video_id || "");
    } catch (err) {
      console.error("Failed to fetch videos:", err);
      if (!silent) setVideosError("Could not connect to the video service.");
    } finally {
      clearTimeout(timeoutId);
      if (!silent) setVideosLoading(false);
    }
  }, []);
  const handleOpenMap = async (videoId) => {
    try {
      const res = await axios.get(`http://localhost:8080/videos/${videoId}/map`);
      setFullMapData(res.data);
      setFullMapSeekTime(null);
      setFullMapHighlightTrack(null);
    } catch (err) {
      console.error("Failed to load map data", err);
      alert("Failed to load map data for this video. Ensure it has processed tracking information.");
    }
  };

  useEffect(() => {
    fetchVideos();
    const retryTimer = setInterval(() => fetchVideos({ silent: true }), 5000);
    return () => clearInterval(retryTimer);
  }, [fetchVideos]);

  useEffect(() => {
    let interval;
    if (processingStatus && processingStatus.status !== 'completed' && processingStatus.status !== 'error') {
      interval = setInterval(async () => {
        try {
          const res = await axios.get(`http://localhost:8080/videos/${activeVideoId}/status`);
          setProcessingStatus(res.data);
          if (res.data.status === 'completed' || res.data.status === 'error' || res.data.status === 'failed') {
            clearInterval(interval);
            fetchVideos();
          }
        } catch(err) {
          console.error(err);
          setProcessingStatus({ status: "failed", progress: 0, stages: ["Analysis failed."] });
          clearInterval(interval);
        }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [processingStatus, activeVideoId, fetchVideos]);

  const handleUpload = async (e) => {
    e.preventDefault();
    setUploadError("");
    if (!videoFile) { setUploadError("Please select a video file first."); return; }
    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", videoFile);
    formData.append("camera_id", cameraId);
    formData.append("location", location);
    try {
      const uploadRes = await axios.post("http://localhost:8080/videos/upload", formData);
      setActiveVideoId(uploadRes.data.video_id);
      setSelectedVideoId(uploadRes.data.video_id);
      await fetchVideos();
      await axios.post(`http://localhost:8080/videos/${uploadRes.data.video_id}/analyze`);
      setIsUploadOpen(false);
      setActiveView("analysis");
      setProcessingStatus({ status: "started", progress: 0, current_step: "Initializing..." });
    } catch(err) {
      console.error(err);
      setUploadError(`Upload failed: ${err.response?.data?.detail || err.message || "Unknown error"}`);
    } finally { setIsUploading(false); }
  };

  const monitorExtendedScans = async (scanInfo, originalQuery) => {
    const scanIds = scanInfo?.scan_ids || [];
    if (scanIds.length === 0) return;
    setExtendedScanStatus({ status: "processing", progress: 0, object: scanInfo.query || originalQuery, scanIds });

    const poll = async () => {
      try {
        const statuses = await Promise.all(
          scanIds.map(scanId => axios.get(`http://localhost:8080/extended-scans/${scanId}`).then(res => res.data))
        );
        const completed = statuses.filter(status => status.status === "completed").length;
        const failed = statuses.filter(status => status.status === "error").length;
        const progress = statuses.length ? statuses.reduce((sum, status) => sum + (status.progress || 0), 0) / statuses.length : 0;
        setExtendedScanStatus({ status: failed ? "error" : completed === statuses.length ? "completed" : "processing", progress, statuses, object: scanInfo.query || originalQuery });

        if (completed + failed === statuses.length) {
          if (failed) return;
          if (scanInfo.map_requested && scanInfo.video_ids?.length === 1) {
            const mapResponse = await axios.get(`http://localhost:8080/videos/${scanInfo.video_ids[0]}/tracks`);
            setFullMapData(mapResponse.data);
            setFullMapSeekTime(null);
            setFullMapHighlightTrack(null);
          } else {
            const searchResponse = await axios.get("http://localhost:8080/search", { params: { q: originalQuery, video_id: selectedVideoId || "ALL" } });
            setMessages(prev => [...prev, {
              role: "assistant",
              content: searchResponse.data.matches?.length
                ? `I found ${searchResponse.data.matches.length} matching extended-object track(s).`
                : "The YOLOE scan completed, but no consistent matching tracks were found.",
              results: searchResponse.data.matches || [],
            }]);
          }
          return;
        }
        setTimeout(poll, 1000);
      } catch (error) {
        console.error("Extended scan status error:", error);
        setExtendedScanStatus({ status: "error", progress: 0, error: error.message });
      }
    };
    poll();
  };

  const sendChat = async (e) => {
    if (e) e.preventDefault();
    if (!chatInput || !chatInput.trim()) return;
    const userInput = chatInput.trim();
    const newUserMsg = { role: "user", content: userInput };
    setMessages(prev => [...prev, newUserMsg]);
    setChatInput("");
    setIsTyping(true);
    try {
      const res = await axios.post("http://localhost:8080/chat", {
          messages: [...messages, newUserMsg],
          video_id: selectedVideoId || "ALL",
          extended_object_detection: extendedObjectDetection,
          session_id: currentSessionId,
      });
      const data = res.data;
      const intent = data.filters?.intent;
      
      setMessages(prev => [...prev, { role: "assistant", content: data.response, results: data.matches }]);
      if (data.session_id) {
        setCurrentSessionId(data.session_id);
        loadChatSessions();
      }
      if (data.extended_scan) {
        setExtendedObjectDetection(true);
        monitorExtendedScans(data.extended_scan, userInput);
      }

      // Handle full_mapping or seek_map intent
      if (intent === "seek_map" || intent === "full_mapping" || data.filters?.full_mapping_data) {
        if (data.filters?.full_mapping_data) {
          setFullMapData(data.filters.full_mapping_data);
          setFullMapSeekTime(data.filters?.time_start || null);
          setFullMapHighlightTrack(data.filters?.track_id || null);
        }
      }
    } catch (err) {
      console.error("Chat error:", err);
      setMessages(prev => [...prev, { role: "assistant", content: "Error connecting to AI backend. Ensure backend is running on port 8080." }]);
    } finally { setIsTyping(false); }
  };
  
  const askWhy = (res) => {
    const contextStr = `Show me why you identified this. [Internal Context: event_id=${res.event_id}]`;
    setChatInput(contextStr);
    setTimeout(() => { document.getElementById('send-btn').click(); }, 100);
  };

  // If full map is open, render exclusively
  if (fullMapData) {
    return (
      <div className="h-screen w-screen">
        <FullVideoMap data={fullMapData} onClose={() => { setFullMapData(null); setFullMapSeekTime(null); setFullMapHighlightTrack(null); }} initialSeekTime={fullMapSeekTime} initialHighlightTrackId={fullMapHighlightTrack} />
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-[#0a0e17] text-white font-sans selection:bg-blue-500/30">
      
      {/* Sidebar */}
      <div className="w-16 md:w-64 border-r border-white/10 bg-[#0f141e] flex flex-col items-center md:items-start p-4 shrink-0 transition-all z-20">
        <div className="flex items-center gap-3 mb-8 text-blue-400 font-bold w-full md:px-2 cursor-pointer" onClick={() => setActiveView("gallery")}>
            <ShieldAlert className="w-8 h-8 shrink-0" />
            <span className="hidden md:inline text-xl tracking-tight text-white">CrimeVision <span className="text-blue-500">AI</span></span>
        </div>
        
        <div className="w-full flex flex-col gap-2 mb-6 text-sm font-semibold">
          <button onClick={() => setActiveView("gallery")} className={`flex items-center gap-3 p-3 rounded-xl transition-colors ${activeView === 'gallery' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:bg-white/5 hover:text-white'}`}>
              <Camera className="w-5 h-5 shrink-0" /> <span className="hidden md:inline">CCTV Gallery</span>
          </button>
          <button onClick={() => setActiveView("analysis")} className={`flex items-center gap-3 p-3 rounded-xl transition-colors ${activeView === 'analysis' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:bg-white/5 hover:text-white'}`}>
              <Activity className="w-5 h-5 shrink-0" /> <span className="hidden md:inline">Video Analysis</span>
          </button>
        </div>

        {activeView === 'analysis' && (
          <div className="w-full flex-1 flex flex-col min-h-0">
            {/* Tabs Header */}
            <div className="flex w-full border-b border-white/10 mb-4 text-xs font-bold gap-2 p-1 bg-black/20 rounded-lg">
              <button 
                onClick={() => setSidebarTab("timeline")} 
                className={`flex-1 flex items-center justify-center gap-2 py-2 px-1 rounded-md transition-all ${sidebarTab === 'timeline' ? 'bg-blue-600 text-white shadow-sm' : 'text-slate-400 hover:text-white'}`}
              >
                <Activity className="w-3.5 h-3.5" />
                <span className="hidden md:inline">Timeline</span>
              </button>
              <button 
                onClick={() => setSidebarTab("history")} 
                className={`flex-1 flex items-center justify-center gap-2 py-2 px-1 rounded-md transition-all ${sidebarTab === 'history' ? 'bg-blue-600 text-white shadow-sm' : 'text-slate-400 hover:text-white'}`}
              >
                <MessageSquare className="w-3.5 h-3.5" />
                <span className="hidden md:inline">Saved Chats</span>
              </button>
            </div>

            {/* Tab content */}
            <div className="flex-1 min-h-0 overflow-y-auto w-full">
              {sidebarTab === 'timeline' ? (
                <IncidentTimeline videoId={activeVideoId} onEventClick={askWhy} />
              ) : (
                <div className="flex flex-col gap-3 h-full">
                  <button 
                    onClick={handleNewChat} 
                    className="w-full flex items-center justify-center gap-2 bg-[#151b2b] hover:bg-[#1f273d] text-white border border-white/10 hover:border-blue-500/30 rounded-xl py-2.5 text-xs font-semibold transition-all shadow-sm"
                  >
                    <PlusCircle className="w-4 h-4 text-blue-400" />
                    <span>+ New Chat Session</span>
                  </button>

                  <div className="flex flex-col gap-2 overflow-y-auto max-h-[50vh] pr-1">
                    {chatSessions.length === 0 ? (
                      <div className="text-center text-xs text-slate-500 py-6">
                        No saved sessions yet.
                      </div>
                    ) : (
                      chatSessions.map((session) => (
                        <div 
                          key={session.session_id} 
                          onClick={() => loadSessionMessages(session.session_id)}
                          className={`group flex items-center justify-between p-3 rounded-xl border text-xs cursor-pointer transition-all ${
                            currentSessionId === session.session_id 
                              ? 'border-blue-500/50 bg-blue-500/10 text-white' 
                              : 'border-white/5 bg-[#151b2b] hover:bg-[#1a2235] text-slate-300'
                          }`}
                        >
                          <div className="flex flex-col gap-1 min-w-0 flex-1">
                            <div className="font-bold truncate pr-2">
                              {session.title || "Untitled Chat"}
                            </div>
                            <div className="text-[10px] text-slate-500 truncate">
                              {new Date(session.created_at).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})} &middot; {session.video_id === 'ALL' ? 'All Videos' : 'Current Video'}
                            </div>
                          </div>
                          <button 
                            onClick={(e) => handleDeleteSession(e, session.session_id)}
                            className="opacity-0 group-hover:opacity-100 hover:text-red-400 p-1 text-slate-500 rounded transition-all"
                            title="Delete Session"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
        
        <div className="mt-auto pt-4 border-t border-white/10 w-full flex justify-center md:justify-start md:px-2">
            <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-full bg-slate-800 flex items-center justify-center shrink-0"><User className="w-4 h-4 text-slate-400" /></div>
                <div className="hidden md:block">
                    <div className="text-xs font-bold">Operator ID: 4921</div>
                    <div className="text-[10px] text-green-400 flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-green-500"></span> Online</div>
                </div>
            </div>
        </div>
      </div>

      {activeView === "gallery" ? (
         <GalleryView 
            videos={availableVideos} 
            fetchVideos={fetchVideos} 
            videosLoading={videosLoading}
            videosError={videosError}
            onUpload={() => { setUploadError(""); setIsUploadOpen(true); }}
            onOpenVideo={(vid) => {
               setSelectedVideoId(vid);
               setActiveVideoId(vid);
               setActiveView("analysis");
            }} 
            onMapVideo={handleOpenMap}
         />
      ) : (
      <div className="flex-1 flex flex-col relative h-full max-h-screen overflow-hidden">
        
        {/* Header */}
        <div className="h-16 border-b border-white/10 flex items-center justify-between px-6 bg-[#0a0e17]/80 backdrop-blur-md sticky top-0 z-10 shrink-0">
          <div className="flex items-center gap-6">
            <h1 className="font-bold text-lg flex items-center gap-2"><Bot className="w-5 h-5 text-blue-500" />AI Investigation Assistant</h1>
            {availableVideos.length > 0 && (
              <div className="flex items-center gap-2 text-sm bg-[#151b2b] px-3 py-1.5 rounded-lg border border-white/10">
                <Video className="w-4 h-4 text-blue-400" />
                <span className="text-slate-400 font-semibold uppercase text-[10px] tracking-wider hidden sm:inline">CCTV SOURCE</span>
                <select className="bg-transparent text-white outline-none font-semibold ml-1 sm:ml-2 sm:border-l border-white/10 sm:pl-2 cursor-pointer appearance-none" value={selectedVideoId} onChange={(e) => setSelectedVideoId(e.target.value)}>
                  <option value="ALL" className="bg-[#151b2b]">ALL CAMERAS</option>
                  {availableVideos.map(v => (
                    <option key={v.video_id} value={v.video_id} className="bg-[#151b2b]">{v.filename} &middot; {v.video_id.substring(0, 8)}</option>
                  ))}
                </select>
              </div>
            )}
            <label className="flex items-center gap-2 text-[10px] uppercase tracking-wider font-bold text-slate-400 bg-[#151b2b] px-3 py-2 rounded-lg border border-white/10 cursor-pointer" title="Run YOLOE only for explicit extended scans or extended mapping">
              <input type="checkbox" checked={extendedObjectDetection} onChange={e => setExtendedObjectDetection(e.target.checked)} className="rounded bg-slate-800 border-white/10 text-emerald-500 focus:ring-0" />
              <span>Extended Object Detection</span>
              <span className={extendedObjectDetection ? "text-emerald-400" : "text-slate-500"}>{extendedObjectDetection ? "ON" : "OFF"}</span>
            </label>
          </div>
          <button onClick={() => setIsUploadOpen(true)} className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white text-sm font-semibold py-2 px-4 rounded-lg transition-colors shadow-[0_0_15px_rgba(37,99,235,0.3)]">
            <UploadCloud className="w-4 h-4" /><span className="hidden sm:inline">+ Upload CCTV</span>
          </button>
        </div>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6 scroll-smooth pb-48">
          {messages.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`flex gap-4 max-w-4xl ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                <div className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 ${msg.role === 'user' ? 'bg-blue-600' : 'bg-slate-800 border border-slate-700'}`}>
                  {msg.role === 'user' ? <User className="w-5 h-5 text-white" /> : <Bot className="w-5 h-5 text-blue-400" />}
                </div>
                <div className="flex flex-col gap-3 min-w-0">
                  <div className={`p-5 rounded-2xl whitespace-pre-wrap leading-relaxed shadow-sm ${msg.role === 'user' ? 'bg-blue-600 text-white rounded-tr-none' : 'bg-[#151b2b] text-slate-200 rounded-tl-none border border-white/5'}`}>
                    {msg.content}
                  </div>
                  {msg.results && msg.results.length > 0 && (<ResultList results={msg.results} onAskWhy={askWhy} />)}
                </div>
              </div>
            </div>
          ))}
          {isTyping && (
             <div className="flex justify-start">
               <div className="flex gap-4 max-w-4xl flex-row">
                 <div className="w-10 h-10 rounded-full flex items-center justify-center shrink-0 bg-slate-800 border border-slate-700"><Bot className="w-5 h-5 text-blue-400" /></div>
                 <div className="p-5 rounded-2xl bg-[#151b2b] text-slate-400 rounded-tl-none border border-white/5 flex items-center gap-2"><Loader2 className="w-4 h-4 animate-spin" /> Analyzing...</div>
               </div>
             </div>
          )}
          <div className="h-40 shrink-0"></div>
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="absolute bottom-0 left-0 w-full p-4 bg-gradient-to-t from-[#0a0e17] via-[#0a0e17] to-transparent pt-12">
          <div className="max-w-4xl mx-auto relative">
            <input type="text" value={chatInput} onChange={e => setChatInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); sendChat(); } }} placeholder="Ask about your CCTV footage (e.g. 'Find the red car', 'show me fully mapped')..." className="w-full bg-[#151b2b] border border-white/10 rounded-full py-4 pl-6 pr-16 text-white focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50 shadow-xl transition-all" />
            <button id="send-btn" onClick={(e) => { e.preventDefault(); sendChat(); }} className="absolute right-2 top-2 bottom-2 w-10 bg-blue-600 hover:bg-blue-500 text-white rounded-full flex items-center justify-center transition-colors"><Send className="w-4 h-4 ml-0.5" /></button>
          </div>
        </div>
      </div>
      )}

      {/* Processing Status */}
      {processingStatus && processingStatus.status !== 'error' && (
        <div className="fixed bottom-6 right-6 w-80 bg-[#151b2b] border border-white/10 rounded-xl shadow-[0_10px_40px_rgba(0,0,0,0.5)] overflow-hidden z-40">
          <div className="p-4 flex flex-col gap-3">
             <div className="flex justify-between items-start mb-3">
                <h4 className="font-bold text-sm text-white flex items-center gap-2">
                   {processingStatus.status === 'completed' ? <CheckCircle2 className="w-4 h-4 text-green-400" /> : processingStatus.status === 'failed' ? <AlertCircle className="w-4 h-4 text-red-400" /> : <Loader2 className="w-4 h-4 animate-spin text-blue-400" />}
                   {processingStatus.status === 'completed' ? 'Analysis Complete' : processingStatus.status === 'failed' ? 'Analysis Failed' : 'Processing CCTV'}
                </h4>
                {processingStatus.status === 'completed' && (<button onClick={() => setProcessingStatus(null)} className="text-slate-400 hover:text-white"><X className="w-4 h-4"/></button>)}
             </div>
             <div className="flex justify-between text-xs text-slate-400 mb-1">
               <span>{processingStatus.status === 'completed' ? 'Done' : processingStatus.status === 'failed' ? 'Failed' : 'Processing'}</span>
               <span>{Math.round(processingStatus.progress)}%</span>
             </div>
             <div className="w-full bg-slate-700 rounded-full h-1.5 overflow-hidden">
               <div className={`h-full transition-all duration-300 ${processingStatus.status === 'failed' ? 'bg-red-500' : 'bg-blue-500'}`} style={{ width: `${Math.round(processingStatus.progress)}%` }}></div>
             </div>
             <div className="text-[10px] text-slate-500 mt-2 truncate">
                {processingStatus.status === 'completed' ? 'Video indexed and ready.' : processingStatus.stages?.[processingStatus.stages.length - 1] || 'Initializing...'}
             </div>
          </div>
        </div>
      )}

      {extendedScanStatus && (
        <div className="fixed bottom-6 left-6 w-96 bg-[#151b2b] border border-emerald-500/30 rounded-xl shadow-[0_10px_40px_rgba(0,0,0,0.5)] overflow-hidden z-40">
          <div className="p-4 flex flex-col gap-3">
            <div className="flex justify-between items-start">
              <h4 className="font-bold text-sm text-white flex items-center gap-2"><Sparkles className="w-4 h-4 text-emerald-400" /> Extended Object Detection</h4>
              {extendedScanStatus.status !== "processing" && <button onClick={() => setExtendedScanStatus(null)} className="text-slate-400 hover:text-white"><X className="w-4 h-4" /></button>}
            </div>
            <div className="flex justify-between text-xs text-slate-400"><span>{extendedScanStatus.status === "completed" ? "YOLOE scan complete" : extendedScanStatus.status === "error" ? "YOLOE scan failed" : "YOLOE scanning"}</span><span>{Math.round(extendedScanStatus.progress || 0)}%</span></div>
            <div className="w-full bg-slate-700 rounded-full h-1.5 overflow-hidden"><div className={`h-full transition-all duration-300 ${extendedScanStatus.status === "error" ? "bg-red-500" : "bg-emerald-500"}`} style={{ width: `${Math.round(extendedScanStatus.progress || 0)}%` }} /></div>
            <div className="text-[10px] text-slate-500 truncate">{extendedScanStatus.status === "completed" ? "Consistent tracks persisted to the shared database." : extendedScanStatus.error || "YOLO11 remains active; YOLOE is query-scoped."}</div>
          </div>
        </div>
      )}

      {/* Upload Modal */}
      {isUploadOpen && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-[#151b2b] border border-white/10 rounded-2xl shadow-2xl w-full max-w-md overflow-hidden relative">
            <div className="flex items-center justify-between p-4 border-b border-white/5">
              <h3 className="font-bold text-lg flex items-center gap-2"><UploadCloud className="w-5 h-5 text-blue-400"/> Upload Footage</h3>
              <button onClick={() => setIsUploadOpen(false)} className="text-slate-400 hover:text-white"><X className="w-5 h-5"/></button>
            </div>
            <form onSubmit={handleUpload} className="p-6 flex flex-col gap-5">
              {uploadError && (<div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 flex items-start gap-2 text-red-400 text-sm"><AlertCircle className="w-4 h-4 mt-0.5 shrink-0" /><p>{uploadError}</p></div>)}
              <div className="border-2 border-dashed border-white/10 hover:border-blue-500/50 rounded-xl p-8 flex flex-col items-center justify-center bg-black/20 transition-colors relative cursor-pointer group">
                <input type="file" accept=".mp4,.mov,.avi" onChange={e => setVideoFile(e.target.files[0])} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" />
                <Video className="w-10 h-10 text-slate-500 group-hover:text-blue-400 mb-3 transition-colors" />
                <span className="text-slate-300 font-semibold text-center">{videoFile ? videoFile.name : "Drop video or Click to Browse"}</span>
              </div>
              <div className="grid grid-cols-2 gap-4">
                 <div className="flex flex-col gap-1.5">
                   <label className="text-xs font-bold text-slate-500 uppercase">Camera</label>
                   <select value={cameraId} onChange={e => setCameraId(e.target.value)} className="bg-black/40 border border-white/10 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:border-blue-500">
                     <option value="CAM_01">CAM_01</option><option value="CAM_02">CAM_02</option><option value="CAM_03">CAM_03</option><option value="CAM_04">CAM_04</option>
                   </select>
                 </div>
                 <div className="flex flex-col gap-1.5">
                   <label className="text-xs font-bold text-slate-500 uppercase">Location</label>
                   <select value={location} onChange={e => setLocation(e.target.value)} className="bg-black/40 border border-white/10 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:border-blue-500">
                     <option value="Main Gate">Main Gate</option><option value="Parking Area">Parking Area</option><option value="North Exit">North Exit</option><option value="Library Road">Library Road</option>
                   </select>
                 </div>
              </div>
              <button type="submit" disabled={isUploading || !videoFile} className="mt-2 w-full bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-500 text-white font-bold py-3.5 rounded-xl flex items-center justify-center gap-2 transition-colors">
                {isUploading ? <><Loader2 className="w-5 h-5 animate-spin" /> Uploading...</> : 'Upload & Analyze'}
              </button>
            </form>
          </div>
        </div>
      )}

    </div>
  );
}
