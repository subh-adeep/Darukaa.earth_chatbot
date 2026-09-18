import React, { useState, useEffect, useRef, useCallback } from 'react';

const PIPELINE_STAGES = [
  { id: 'parse',   icon: '🔍', label: 'Parsing environmental context',         sub: 'Extracting soil, climate, and land-use variables from your query…' },
  { id: 'plan',    icon: '🧠', label: 'Generating research hypotheses',          sub: 'LLM #1 forming targeted multi-metric retrieval queries…' },
  { id: 'retrieve',icon: '📚', label: 'Retrieving evidence from knowledge base', sub: 'Semantic search across 7,978 peer-reviewed passages via Qdrant…' },
  { id: 'reason',  icon: '⚗️',  label: 'Synthesising multi-metric reasoning',    sub: 'LLM #2 (Gemini) grounding recommendations in retrieved evidence…' },
  { id: 'format',  icon: '✅', label: 'Formatting science-backed plan',          sub: 'Structuring interventions, trade-offs, and evidence citations…' },
];

async function callGradioEndpoint(endpoint, inputData) {
  const postRes = await fetch(`https://subhadeepsing-drakula.hf.space/gradio_api/call/${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data: [inputData] })
  });
  if (!postRes.ok) throw new Error(`API error: ${postRes.statusText}`);
  const { event_id } = await postRes.json();

  const streamRes = await fetch(`https://subhadeepsing-drakula.hf.space/gradio_api/call/${endpoint}/${event_id}`);
  if (!streamRes.body) throw new Error('ReadableStream not supported in this browser.');

  const reader = streamRes.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      if (!line.startsWith('data:')) continue;
      const jsonStr = line.slice(5).trim();
      if (!jsonStr || jsonStr === 'null') continue;
      try {
        const arr = JSON.parse(jsonStr);
        if (arr && arr[0] !== undefined) {
          try { reader.cancel(); } catch { }
          return typeof arr[0] === 'string' ? JSON.parse(arr[0]) : arr[0];
        }
      } catch (e) {
        // continue buffering
      }
    }
  }
  throw new Error("No response received from model");
}

async function fetchRecommend(payload) {
  if (typeof window !== 'undefined' && window.location.hostname === 'localhost' && window.location.port === '5173') {
    const res = await fetch("http://localhost:8000/api/recommend", {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Server error'); }
    return await res.json();
  }

  const payloadStr = JSON.stringify({
    text: payload.text || '',
    conversation_history: payload.conversation_history || [],
    structured: payload.structured || null
  });
  return await callGradioEndpoint('recommend_json', payloadStr);
}

async function fetchRetrieve(query, mode = 'semantic', top_k = 25, min_score = 0.40) {
  if (typeof window !== 'undefined' && window.location.hostname === 'localhost' && window.location.port === '5173') {
    const res = await fetch("http://localhost:8000/api/retrieve", {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, mode, top_k, min_score })
    });
    return await res.json();
  }

  const payloadStr = JSON.stringify({ query, mode, top_k, min_score });
  return await callGradioEndpoint('retrieve_json', payloadStr);
}


const STARTER_PROMPTS = [
  {
    title: "Semi-Arid Wheat Monoculture",
    desc: "0.3% SOC · < 380mm rainfall · low pollinators",
    text: "Biodiversity is declining on my land. We grow continuous monoculture wheat in a semi-arid zone with 0.3% soil organic carbon and under 380mm annual rainfall."
  },
  {
    title: "Mediterranean Hillside Vineyard",
    desc: "Severe topsoil erosion · low understory diversity",
    text: "Our Mediterranean vineyard suffers from severe topsoil erosion and 0.7% soil organic matter. Tillage has eliminated ground cover, resulting in high runoff and negligible pollinator diversity."
  },
  {
    title: "Tropical Acidic Maize Farm",
    desc: "Heavy leaching · pH 5.2 · high pest pressure",
    text: "We farm maize on degraded sandy loam in a tropical wet-dry zone. Soil organic carbon is 0.6%, rainfall causes leaching, and pest pressure is extreme due to loss of predatory insects."
  },
  {
    title: "Overgrazed Dryland Pasture",
    desc: "Soil compaction · loss of perennial grasses",
    text: "Continuous cattle grazing on our semi-arid rangeland has caused soil compaction, poor water infiltration, and loss of native perennial grasses. Soil carbon is 0.4%."
  }
];

const FORM_SUGGESTIONS = {
  region: ["Semi-Arid", "Mediterranean", "Tropical Wet-Dry", "Temperate Grassland", "Arid Rangeland", "Humid Subtropical"],
  crop: ["Wheat", "Maize", "Grapevine", "Rice", "Barley", "Cotton", "Sorghum", "Soybean"],
  land_use: ["Monoculture", "Conventional Tillage", "No-Till Conservation", "Crop Rotation", "Agroforestry", "Intensive Pasture"],
  soil_organic_carbon: ["0.3", "0.5", "0.8", "1.2", "1.8", "2.5"],
  rainfall: ["350", "450", "650", "980", "1200"],
  soil_moisture: ["Sandy Loam (Dry)", "Clay Loam", "Compacted Topsoil", "Well-Drained Loam"],
  habitat_diversity: ["Severe Decline", "Low Pollinators", "Moderate / Patchy", "Degraded Remnants"],
  other_useful_info: ["Soil crusting & low earthworms", "High pest pressure, no predators", "Historical chemical fertilizer use", "Steep erosion-prone slope"]
};

// Typewriter component — animates any text character by character
function TypewriterText({ text, speed = 8, onFinished }) {
  const [displayed, setDisplayed] = useState('');
  const doneRef = useRef(false);
  const prevTextRef = useRef('');

  useEffect(() => {
    if (!text) { setDisplayed(''); return; }
    if (text === prevTextRef.current) return; // avoid re-running on same text
    prevTextRef.current = text;
    doneRef.current = false;
    setDisplayed('');
    let idx = 0;
    const iv = setInterval(() => {
      idx += 3; // advance 3 chars at a time for speed
      if (idx >= text.length) {
        setDisplayed(text);
        clearInterval(iv);
        if (!doneRef.current) { doneRef.current = true; onFinished?.(); }
      } else {
        setDisplayed(text.slice(0, idx));
      }
    }, speed);
    return () => clearInterval(iv);
  }, [text, speed]);

  return (
    <span className="typewriter-wrap">
      {displayed}
      {!doneRef.current && displayed.length < (text?.length || 0) && <span className="typing-cursor" />}
    </span>
  );
}

// Renders a plain text response (greeting / off-topic / follow-up) with basic markdown
// bold (**text**) and newlines, with optional typewriter
function SimpleMessage({ text, animate, onFinished }) {
  const formatted = (text || '').replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br/>');
  if (animate) {
    return (
      <TypewriterText
        text={text}
        onFinished={onFinished}
        speed={6}
      />
    );
  }
  return <span dangerouslySetInnerHTML={{ __html: formatted }} />;
}

// Inline citation badge renderer: converts [EXXXX] patterns to styled badges
function CitationText({ text }) {
  if (!text) return null;
  const parts = text.split(/(\[E\d{1,5}\])/gi);
  return (
    <span>
      {parts.map((part, i) => {
        if (/^\[E\d{1,5}\]$/i.test(part)) {
          return (
            <span key={i} className="inline-citation">{part}</span>
          );
        }
        return part;
      })}
    </span>
  );
}

// Renders a full recommendation as a clean ChatGPT-like report
function RecommendationReport({ msg, animate, onFinished }) {
  const [typeDone, setTypeDone] = useState(!animate);

  // Build a full text string for the typewriter effect
  // We render the full markdown-like text as a single stream
  const sections = [];

  if (msg.primary_intervention) {
    sections.push(`## Recommendation\n${msg.primary_intervention}`);
  }
  if (msg.environmental_assessment) {
    sections.push(`## Ecosystem Assessment\n${msg.environmental_assessment}`);
  }

  const recTexts = (msg.recommendations || []).map((rec, ri) => {
    let s = `## Intervention ${ri + 1}: ${rec.what_to_do}`;
    if (rec.why_it_works) s += `\n**Why it works:** ${rec.why_it_works}`;
    if (rec.measurable_outcome) s += `\n**Impact:** ${rec.measurable_outcome}`;
    if (rec.impacted_metrics?.length) s += `\n**Metrics:** ${rec.impacted_metrics.join(' · ')}`;
    if (rec.time_horizon) s += `\n**Time Horizon:** ${rec.time_horizon}`;
    if (rec.reference_citation) s += `\n**Reference:** ${rec.reference_citation}`;
    return s;
  });
  if (recTexts.length) sections.push(recTexts.join('\n\n'));

  if (msg.trade_offs?.length) {
    const tlist = msg.trade_offs.map(t => `• ${t.claim || t}`).join('\n');
    sections.push(`## Trade-offs & Mitigations\n${tlist}`);
  }

  if (msg.retrieved_evidence?.length) {
    const evList = msg.retrieved_evidence.slice(0, 5).map(ev =>
      `[${ev.evidence_id}] ${ev.doc_title}: "${ev.text?.slice(0, 120)}…"`
    ).join('\n');
    sections.push(`## Evidence References\n${evList}`);
  }

  const fullText = sections.join('\n\n');

  const [rendered, setRendered] = useState(animate ? '' : fullText);

  useEffect(() => {
    if (!animate) { setRendered(fullText); setTypeDone(true); return; }
    setRendered('');
    setTypeDone(false);
    let idx = 0;
    const iv = setInterval(() => {
      idx += 4;
      if (idx >= fullText.length) {
        setRendered(fullText);
        clearInterval(iv);
        setTypeDone(true);
        onFinished?.();
      } else {
        setRendered(fullText.slice(0, idx));
      }
    }, 6);
    return () => clearInterval(iv);
  }, [animate, fullText]);

  // Parse rendered text into structured display
  return (
    <div className="chat-report">
      {msg.data_availability_notice && (
        <div className="report-notice">
          <span>📍</span>
          <span>{msg.data_availability_notice}</span>
        </div>
      )}
      <ParsedReport text={rendered} isDone={typeDone} />
      {!typeDone && <span className="typing-cursor" />}
    </div>
  );
}

// Helper to extract follow-up questions from text if not separately delivered
function extractFollowUpQuestions(text) {
  if (!text) return [];
  const match = text.match(/(?:#{1,3}\s*|\*{1,2}|)?Explore Further/i);
  if (!match) return [];
  const section = text.slice(match.index + match[0].length);
  const rawItems = section.split(/(?:^|\n|\s+)\d+[\.\)]\s+/);
  return rawItems
    .map(q => q.trim().replace(/^[:\-\s]+/, '').trim())
    .filter(q => q.length > 10 && !q.toLowerCase().startsWith('explore further'))
    .map(q => q.endsWith('?') ? q : q + '?')
    .slice(0, 4);
}

// Helper to strip Explore Further block from raw report body
function stripExploreFurther(text) {
  if (!text) return '';
  const match = text.match(/(?:#{1,3}\s*|\*{1,2}|)?Explore Further/i);
  if (!match) return text;
  return text.slice(0, match.index).trim();
}

// Parses the progressive text into rendered sections
function ParsedReport({ text, isDone }) {
  const cleanText = isDone ? stripExploreFurther(text) : text;
  const lines = (cleanText || '').split('\n');
  const elements = [];
  let buffer = [];
  let keyIdx = 0;

  const flushBuffer = () => {
    if (buffer.length) {
      const joined = buffer.join('\n').trim();
      if (joined) {
        elements.push(
          <p key={keyIdx++} className="report-para">
            <CitationText text={joined} />
          </p>
        );
      }
      buffer = [];
    }
  };

  for (const line of lines) {
    // If during streaming we reach Explore Further, do not render it in body
    if (!isDone && line.match(/(?:#{1,3}\s*|\*{1,2}|)?Explore Further/i)) {
      break;
    }
    if (line.startsWith('## ')) {
      flushBuffer();
      elements.push(
        <h2 key={keyIdx++} className="report-heading">{line.slice(3)}</h2>
      );
    } else if (line.startsWith('**') && line.endsWith('**')) {
      flushBuffer();
      elements.push(
        <p key={keyIdx++} className="report-label">{line.slice(2, -2)}</p>
      );
    } else if (line.match(/^\*\*(.+?):\*\*/)) {
      flushBuffer();
      const match = line.match(/^\*\*(.+?):\*\*(.*)/);
      if (match) {
        elements.push(
          <p key={keyIdx++} className="report-field">
            <span className="report-field-key">{match[1]}:</span>
            <span> <CitationText text={match[2].trim()} /></span>
          </p>
        );
      } else {
        buffer.push(line);
      }
    } else if (line.match(/^([A-Za-z\s]+):\s*$/)) {
      // Standalone label like "Overall Confidence:" or "Trade-offs & Mitigations:"
      flushBuffer();
      elements.push(
        <p key={keyIdx++} className="report-field">
          <span className="report-field-key">{line.replace(':', '').trim()}:</span>
        </p>
      );
    } else if (line.startsWith('• ') || line.startsWith('* ') || line.startsWith('- ')) {
      flushBuffer();
      elements.push(
        <div key={keyIdx++} className="report-bullet">
          <span>•</span>
          <span><CitationText text={line.slice(2)} /></span>
        </div>
      );
    } else if (line.match(/^\d+[\.\)]\s+/)) {
      flushBuffer();
      const match = line.match(/^(\d+[\.\)])\s+(.*)/);
      if (match) {
        elements.push(
          <div key={keyIdx++} className="report-bullet">
            <span style={{fontFamily:'var(--mono)',color:'var(--green)',fontSize:12,marginRight:4}}>{match[1]}</span>
            <span><CitationText text={match[2]} /></span>
          </div>
        );
      } else {
        buffer.push(line);
      }
    } else {
      buffer.push(line);
    }
  }
  flushBuffer();

  return <div className="report-body">{elements}</div>;
}

// SVG Icons (inline, no external dep needed beyond lucide)
const Icon = {
  Leaf: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10z"/>
      <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>
    </svg>
  ),
  Bot: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect width="18" height="10" x="3" y="11" rx="2"/>
      <circle cx="12" cy="5" r="2"/>
      <path d="M12 7v4"/>
      <line x1="8" x2="8" y1="16" y2="16"/>
      <line x1="16" x2="16" y1="16" y2="16"/>
    </svg>
  ),
  User: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="8" r="4"/>
      <path d="M20 21a8 8 0 0 0-16 0"/>
    </svg>
  ),
  Send: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m22 2-7 20-4-9-9-4Z"/>
      <path d="M22 2 11 13"/>
    </svg>
  ),
  Refresh: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>
      <path d="M21 3v5h-5"/>
      <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>
      <path d="M8 16H3v5"/>
    </svg>
  ),
  Search: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8"/>
      <path d="m21 21-4.3-4.3"/>
    </svg>
  ),
  Database: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="12" cy="5" rx="9" ry="3"/>
      <path d="M3 5V19A9 3 0 0 0 21 19V5"/>
      <path d="M3 12A9 3 0 0 0 21 12"/>
    </svg>
  ),
  Sliders: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="4" x2="4" y1="21" y2="14"/>
      <line x1="4" x2="4" y1="10" y2="3"/>
      <line x1="12" x2="12" y1="21" y2="12"/>
      <line x1="12" x2="12" y1="8" y2="3"/>
      <line x1="20" x2="20" y1="21" y2="16"/>
      <line x1="20" x2="20" y1="12" y2="3"/>
      <line x1="2" x2="6" y1="14" y2="14"/>
      <line x1="10" x2="14" y1="8" y2="8"/>
      <line x1="18" x2="22" y1="16" y2="16"/>
    </svg>
  ),
  Zap: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>
    </svg>
  ),
  Clock: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <polyline points="12 6 12 12 16 14"/>
    </svg>
  ),
  ChevronDown: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m6 9 6 6 6-6"/>
    </svg>
  ),
  ChevronUp: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m18 15-6-6-6 6"/>
    </svg>
  ),
  Alert: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
      <path d="M12 9v4"/>
      <path d="M12 17h.01"/>
    </svg>
  ),
  Book: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>
    </svg>
  ),
  Shield: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>
    </svg>
  ),
  Sparkles: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/>
    </svg>
  ),
  Help: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
      <path d="M12 17h.01"/>
    </svg>
  ),
};

export default function App() {
  const [activeTab, setActiveTab] = useState('chat');

  // Chat state
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [chatError, setChatError] = useState(null);
  const [openEvidence, setOpenEvidence] = useState({});
  const chatBottomRef = useRef(null);

  // Scenario state
  const [scenarioForm, setScenarioForm] = useState({
    region: 'Semi-Arid',
    crop: 'Wheat',
    land_use: 'Monoculture',
    soil_organic_carbon: '0.3',
    rainfall: '450',
    soil_moisture: 'Sandy Loam (Dry)',
    habitat_diversity: 'Severe Decline',
    other_useful_info: ''
  });
  const [scenarioLoading, setScenarioLoading] = useState(false);
  const [scenarioResult, setScenarioResult] = useState(null);
  const [scenarioError, setScenarioError] = useState(null);

  // Explorer state
  const [searchQuery, setSearchQuery] = useState('cover crops soil organic carbon water retention');
  const [searchMode, setSearchMode] = useState('semantic'); // 'semantic' | 'id'
  const [visibleCount, setVisibleCount] = useState(5);
  const [searchResults, setSearchResults] = useState(null);
  const [isSearching, setIsSearching] = useState(false);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping, isLoading]);

  const toggleEvidence = (id) => setOpenEvidence(p => ({ ...p, [id]: !p[id] }));

  // Gradio SSE stream consumer — reads token-by-token events from the stream_recommend endpoint
  const fetchRecommendStream = async (payload, onEvent) => {
    const IS_LOCAL = window.location.hostname === 'localhost' && window.location.port === '5173';
    const SPACE_BASE = 'https://subhadeepsing-drakula.hf.space';
    const payloadStr = JSON.stringify({ text: payload.text || '', conversation_history: payload.conversation_history || [] });

    if (IS_LOCAL) {
      // Localhost: call legacy endpoint, simulate streaming events
      const res = await fetch('http://localhost:8000/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      onEvent({ event: 'stage', stage: 'parse' });
      await new Promise(r => setTimeout(r, 200));
      if (data.is_greeting) { onEvent({ event: 'done', type: 'greeting', reply: data.reply || data.greeting_reply, timing: data.timing }); return; }
      if (data.is_off_topic) { onEvent({ event: 'done', type: 'off_topic', reply: data.reply || data.off_topic_reply, timing: data.timing }); return; }
      if (data.is_follow_up) { onEvent({ event: 'done', type: 'follow_up', reply: data.reply, timing: data.timing }); return; }
      if (data.needs_clarification && data.clarification) { onEvent({ event: 'done', type: 'clarification', clarification: data.clarification, timing: data.timing }); return; }
      if (data.recommendation) { onEvent({ event: 'done', type: 'recommendation', text: data.recommendation.primary_intervention, title: data.recommendation.title, follow_up_questions: [], retrieved_evidence: data.recommendation.retrieved_evidence || [], timing: data.timing }); return; }
      onEvent({ event: 'done', type: 'greeting', reply: data.reply || JSON.stringify(data), timing: {} });
      return;
    }

    // HF Space: full SSE streaming via Gradio generator
    const postRes = await fetch(`${SPACE_BASE}/gradio_api/call/stream_recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: [payloadStr] })
    });
    if (!postRes.ok) throw new Error(`Stream API POST failed: ${postRes.statusText}`);
    const { event_id } = await postRes.json();

    const streamRes = await fetch(`${SPACE_BASE}/gradio_api/call/stream_recommend/${event_id}`);
    if (!streamRes.body) throw new Error('ReadableStream not supported in this browser.');

    const reader = streamRes.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (!raw || raw === 'null') continue;
        try {
          const arr = JSON.parse(raw);
          if (!arr || !arr[0]) continue;
          const ev = typeof arr[0] === 'string' ? JSON.parse(arr[0]) : arr[0];
          if (ev && ev.event) onEvent(ev);
        } catch { /* skip malformed SSE line */ }
      }
    }
  };

  const handleNewChat = () => {
    setMessages([]); setChatInput(''); setIsTyping(false);
    setIsLoading(false); setChatError(null); setOpenEvidence({});
  };

  // ── Streaming message sender ──────────────────────────────────────────────
  const handleSendMessage = async (textToSend) => {
    const text = (textToSend ?? chatInput).trim();
    if (!text || isLoading || isTyping) return;

    const historyPayload = messages.map(m => ({
      role: m.role,
      content: m.role === 'user' ? m.content : (m.summaryText || m.content || '')
    }));

    const userMsgId = `usr_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const streamMsgId = `asst_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    setMessages(prev => [
      ...prev,
      { id: userMsgId, role: 'user', content: text },
      // Placeholder streaming message — transitions to final type on done event
      { id: streamMsgId, role: 'assistant', type: 'streaming', stage: 0, showText: false, accumulated: '', summaryText: '' }
    ]);
    setChatInput('');
    setIsLoading(true);
    setChatError(null);

    try {
      await fetchRecommendStream(
        { text, conversation_history: historyPayload },
        (ev) => {
          if (ev.event === 'stage') {
            const stageMap = { parse: 0, plan: 1, retrieve: 2, reason: 3, format: 4 };
            const si = stageMap[ev.stage] ?? 0;
            setMessages(prev => prev.map(m =>
              m.id === streamMsgId && m.role === 'assistant'
                ? { ...m, stage: si }
                : m
            ));
          } else if (ev.event === 'token') {
            setMessages(prev => prev.map(m =>
              m.id === streamMsgId && m.role === 'assistant'
                ? { ...m, accumulated: ev.accumulated || '', showText: true, stage: Math.max(m.stage, 3) }
                : m
            ));
          } else if (ev.event === 'done') {
            const { type, reply, clarification, timing } = ev;
            if (type === 'greeting') {
              setMessages(prev => prev.map(m =>
                m.id === streamMsgId && m.role === 'assistant' ? { ...m, type: 'greeting', content: reply, timing, summaryText: reply } : m
              ));
              setIsTyping(true);
            } else if (type === 'off_topic') {
              setMessages(prev => prev.map(m =>
                m.id === streamMsgId && m.role === 'assistant' ? { ...m, type: 'off_topic', content: reply, timing, summaryText: reply } : m
              ));
              setIsTyping(true);
            } else if (type === 'clarification') {
              setMessages(prev => prev.map(m =>
                m.id === streamMsgId && m.role === 'assistant' ? {
                  ...m, type: 'clarification',
                  content: clarification?.clarifying_question,
                  missing_variables: clarification?.missing_variables || [],
                  extracted_so_far: clarification?.extracted_so_far || {},
                  timing, summaryText: clarification?.clarifying_question
                } : m
              ));
              setIsTyping(true);
            } else if (type === 'follow_up') {
              // Text already streamed token-by-token; just finalize
              setMessages(prev => prev.map(m =>
                m.id === streamMsgId && m.role === 'assistant' ? {
                  ...m, type: 'follow_up',
                  content: ev.text || reply || m.accumulated,
                  timing, summaryText: 'Follow-up'
                } : m
              ));
            } else if (type === 'recommendation') {
              // Report fully streamed; add metadata (follow-ups, evidence)
              setMessages(prev => prev.map(m =>
                m.id === streamMsgId && m.role === 'assistant' ? {
                  ...m, type: 'recommendation_text',
                  content: ev.text || m.accumulated,
                  title: ev.title || 'Ecological Intervention Plan',
                  follow_up_questions: ev.follow_up_questions || [],
                  retrieved_evidence: ev.retrieved_evidence || [],
                  validation_flags: ev.validation_flags || [],
                  data_availability_notice: ev.data_availability_notice,
                  timing, summaryText: ev.title || 'Ecological Intervention Plan'
                } : m
              ));
            } else {
              setMessages(prev => prev.map(m =>
                m.id === streamMsgId && m.role === 'assistant' ? { ...m, type: 'greeting', content: reply || '', timing, summaryText: reply || '' } : m
              ));
              setIsTyping(true);
            }
            setIsLoading(false);
          }
        }
      );
    } catch (err) {
      setChatError(err.message);
      setMessages(prev => prev.filter(m => m.id !== streamMsgId));
      setIsLoading(false);
    }
  };

  const handleScenarioSubmit = async (e) => {
    e.preventDefault();
    setScenarioLoading(true); setScenarioError(null); setScenarioResult(null);

    // Validate mandatory fields
    const mandatoryFields = [
      { key: 'region', label: 'Ecological Region' },
      { key: 'crop', label: 'Current Crop' },
      { key: 'land_use', label: 'Land Use / Farming Practice' },
      { key: 'soil_organic_carbon', label: 'Soil Organic Carbon (%)' },
      { key: 'rainfall', label: 'Annual Rainfall (mm)' },
      { key: 'soil_moisture', label: 'Soil Moisture & Texture' },
      { key: 'habitat_diversity', label: 'Habitat Diversity' }
    ];

    const missing = mandatoryFields.filter(f => !String(scenarioForm[f.key] || '').trim());
    if (missing.length > 0) {
      setScenarioError(`All fields except "Other Information" are mandatory. Please provide: ${missing.map(m => m.label).join(', ')}.`);
      setScenarioLoading(false);
      return;
    }

    // Number validation for SOC
    let socVal = parseFloat(scenarioForm.soil_organic_carbon);
    if (isNaN(socVal) || socVal <= 0) {
      setScenarioError("Soil Organic Carbon (%) must be a valid positive number (digits only, e.g. 0.3 or 1.2).");
      setScenarioLoading(false);
      return;
    }

    // Number validation for Rainfall
    let rainVal = parseInt(scenarioForm.rainfall, 10);
    if (isNaN(rainVal) || rainVal <= 0) {
      setScenarioError("Annual Rainfall must be a valid integer in millimeters (digits only, e.g. 450 or 980).");
      setScenarioLoading(false);
      return;
    }

    try {
      const otherNotes = (scenarioForm.other_useful_info || '').trim();
      const queryText = `In ${scenarioForm.region} region under ${scenarioForm.crop} cultivation with ${scenarioForm.land_use} farming practice, soil organic carbon of ${socVal}%, soil moisture and texture of ${scenarioForm.soil_moisture}, annual rainfall of ${rainVal} mm, and habitat diversity status of ${scenarioForm.habitat_diversity}${otherNotes ? `, observed site features: ${otherNotes}` : ''}, formulate an evidence-grounded ecological intervention.`;

      // Seamlessly integrate with the live streaming Chat pipeline
      setScenarioError(null);
      setActiveTab('chat');
      handleSendMessage(queryText);
    } catch (err) {
      setScenarioError(err.message);
    } finally {
      setScenarioLoading(false);
    }
  };

  const handleExplorerSearch = async (e, overrideQuery, overrideMode) => {
    e?.preventDefault();
    const q = (overrideQuery ?? searchQuery).trim();
    const mode = overrideMode ?? searchMode;
    if (!q || isSearching) return;
    setIsSearching(true);
    setVisibleCount(5);
    try {
      const data = await fetchRetrieve(q, mode, 25, 0.40);
      setSearchResults(data);
    } catch (err) { console.error(err); }
    finally { setIsSearching(false); }
  };

  const sf = (key, val) => setScenarioForm(p => ({ ...p, [key]: val }));

  return (
    <div className="app">
      {/* ── Navigation ─────────────────────────────────────────────── */}
      <nav className="nav">
        <div className="nav-brand">
          <div className="nav-logo">🌿</div>
          <div>
            <div className="nav-name">Darukaa.Earth</div>
            <div className="nav-sub">AI Biodiversity Intelligence · 7,978 Verified Passages</div>
          </div>
        </div>

        <div className="nav-right">
          <div className="tab-bar">
            <button className={`tab-btn${activeTab === 'chat' ? ' active' : ''}`} onClick={() => setActiveTab('chat')}>
              <span style={{width:14,height:14,display:'inline-flex'}}><Icon.Bot/></span>
              <span>AI Chat</span>
            </button>
            <button className={`tab-btn${activeTab === 'scenario' ? ' active' : ''}`} onClick={() => setActiveTab('scenario')}>
              <span style={{width:14,height:14,display:'inline-flex'}}><Icon.Sliders/></span>
              <span>Planner</span>
            </button>
            <button className={`tab-btn${activeTab === 'explorer' ? ' active' : ''}`} onClick={() => setActiveTab('explorer')}>
              <span style={{width:14,height:14,display:'inline-flex'}}><Icon.Database/></span>
              <span>Evidence</span>
            </button>
          </div>

          {activeTab === 'chat' && (
            <button className="new-chat-btn" onClick={handleNewChat} title="Start new conversation">
              <span style={{width:13,height:13,display:'inline-flex'}}><Icon.Refresh/></span>
              <span>New Chat</span>
            </button>
          )}
        </div>
      </nav>

      {/* ── Content ────────────────────────────────────────────────── */}
      <div className="content">

        {/* ── TAB: CHAT ─────────────────────────────────────────────── */}
        {activeTab === 'chat' && (
          <div className="chat-pane">
            <div className="chat-messages">
              {/* Empty State */}
              {messages.length === 0 && (
                <div className="empty-state">
                  <div className="empty-icon">🌱</div>
                  <div>
                    <div className="empty-title">Evidence-Grounded Environmental AI</div>
                    <div className="empty-sub" style={{marginTop:8}}>
                      Describe your ecosystem conditions or land management challenge. Darukaa AI reasons across soil carbon, moisture dynamics, cropping systems, and biodiversity using 7,978 verified scientific passages.
                    </div>
                  </div>

                  {/* Planner Suggestion Card */}
                  <div className="planner-callout-card" onClick={() => setActiveTab('scenario')}>
                    <div className="planner-callout-left">
                      <div className="planner-callout-badge">💡 Pro Tip</div>
                      <div className="planner-callout-title">Need a targeted multi-variable farm report?</div>
                      <div className="planner-callout-desc">
                        Use the <strong>Scenario Planner</strong> tab to dial in exact soil organic carbon, rainfall, crop, and ecological constraints in one click.
                      </div>
                    </div>
                    <button type="button" className="planner-callout-btn" onClick={(e) => { e.stopPropagation(); setActiveTab('scenario'); }}>
                      <span>Try Planner</span>
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{width:13,height:13}}>
                        <path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>
                      </svg>
                    </button>
                  </div>

                  <div className="starter-grid">
                    {STARTER_PROMPTS.map((p, i) => (
                      <button key={i} className="starter-card" onClick={() => setChatInput(p.text)}>
                        <div className="starter-title">{p.title}</div>
                        <div className="starter-desc">{p.desc}</div>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Messages */}
              {messages.map((msg, idx) => {
                const isLatest = idx === messages.length - 1;

                if (msg.role === 'user') return (
                  <div key={msg.id} className="msg-row user">
                    <div className="msg-avatar user">👤</div>
                    <div className="msg-body">
                      <div className="bubble-user">{msg.content}</div>
                    </div>
                  </div>
                );

                if (msg.type === 'greeting') return (
                  <div key={msg.id} className="msg-row">
                    <div className="msg-avatar ai">🌿</div>
                    <div className="msg-body">
                      <div className="bubble-ai simple-bubble">
                        <SimpleMessage
                          text={msg.content}
                          animate={isLatest && isTyping}
                          onFinished={() => setIsTyping(false)}
                        />
                        {msg.timing && (
                          <div className="msg-timing">⚡ {msg.timing.total_time_s ?? 0.3}s</div>
                        )}
                      </div>
                    </div>
                  </div>
                );

                if (msg.type === 'off_topic') return (
                  <div key={msg.id} className="msg-row">
                    <div className="msg-avatar ai">🌿</div>
                    <div className="msg-body">
                      <div className="bubble-ai simple-bubble" style={{borderLeft:'3px solid #64748b'}}>
                        <SimpleMessage
                          text={msg.content}
                          animate={isLatest && isTyping}
                          onFinished={() => setIsTyping(false)}
                        />
                        {msg.timing && (
                          <div className="msg-timing">⚡ {msg.timing.total_time_s ?? 0.3}s</div>
                        )}
                      </div>
                    </div>
                  </div>
                );

                // follow_up — streamed markdown rendered with ParsedReport
                if (msg.type === 'follow_up') return (
                  <div key={msg.id} className="msg-row">
                    <div className="msg-avatar ai">🌿</div>
                    <div className="msg-body" style={{flex:1, minWidth:0}}>
                      <div className="bubble-ai rec-bubble">
                        <div className="chat-report">
                          <ParsedReport text={msg.content || ''} isDone={true} />
                        </div>
                        {msg.timing && <div className="msg-timing">⚡ {msg.timing.total_time_s ?? 0.3}s</div>}
                      </div>
                    </div>
                  </div>
                );

                // recommendation_text — fully streamed markdown report + evidence + sleek follow-up hub
                if (msg.type === 'recommendation_text') {
                  const followUps = (msg.follow_up_questions && msg.follow_up_questions.length > 0)
                    ? msg.follow_up_questions
                    : extractFollowUpQuestions(msg.content);

                  return (
                    <div key={msg.id} className="msg-row">
                      <div className="msg-avatar ai">🌿</div>
                      <div className="msg-body" style={{flex:1, minWidth:0}}>
                        <div className="bubble-ai rec-bubble">
                          <div className="rec-title-row">
                            <span className="rec-main-title"><span style={{marginRight:8}}>🌿</span>{msg.title}</span>
                            {msg.timing && (
                              <div className="timing-row">
                                <span className="timing-pill green"><span style={{width:10,height:10,display:'inline-flex'}}><Icon.Zap/></span>{msg.timing.total_time_s ?? 2.1}s</span>
                                <span className="timing-pill cyan"><span style={{width:10,height:10,display:'inline-flex'}}><Icon.Search/></span>{msg.timing.retrieval_time_s ?? 0.35}s retrieval</span>
                                <span className="timing-pill purple"><span style={{width:10,height:10,display:'inline-flex'}}><Icon.Clock/></span>{msg.timing.thinking_time_s ?? 1.75}s reasoning</span>
                              </div>
                            )}
                          </div>
                          <div className="report-divider" />
                          {msg.data_availability_notice && (
                            <div className="report-notice">
                              <span>📍</span><span>{msg.data_availability_notice}</span>
                            </div>
                          )}
                          <div className="chat-report">
                            <ParsedReport text={msg.content || ''} isDone={true} />
                          </div>

                          {/* Evidence drawer */}
                          {msg.retrieved_evidence?.length > 0 && (
                            <div style={{marginTop:16}}>
                              <button className="evidence-toggle" onClick={() => toggleEvidence(msg.id)}>
                                <span style={{display:'flex',alignItems:'center',gap:6}}>
                                  <span style={{width:12,height:12,display:'inline-flex',color:'var(--green)'}}><Icon.Book/></span>
                                  View {msg.retrieved_evidence.length} peer-reviewed evidence passages
                                </span>
                                <span style={{width:14,height:14,display:'inline-flex'}}>
                                  {openEvidence[msg.id] ? <Icon.ChevronUp/> : <Icon.ChevronDown/>}
                                </span>
                              </button>
                              {openEvidence[msg.id] && (
                                <div className="evidence-list">
                                  {msg.retrieved_evidence.map((ev, ei) => (
                                    <div key={ei} className="evidence-chunk">
                                      <div className="ev-meta">
                                        <span className="ev-id">[{ev.evidence_id}]</span>
                                        <span className="ev-doc">{ev.doc_title}</span>
                                        <span className="ev-score">score {(ev.score || 0).toFixed(3)}</span>
                                      </div>
                                      <div className="ev-text">"{ev.text}"</div>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          )}

                          {/* Sleek Follow-up Investigation Hub */}
                          {followUps?.length > 0 && (
                            <div className="followup-container">
                              <div className="followup-header">
                                <div className="followup-title-wrap">
                                  <span className="followup-icon-glow">✨</span>
                                  <span className="followup-heading">Suggested Inquiries & Next Steps</span>
                                </div>
                                <span className="followup-badge">Click to customize · or send directly</span>
                              </div>
                              <div className="followup-grid">
                                {followUps.map((q, qi) => (
                                  <div key={qi} className="followup-card">
                                    <div className="followup-card-body" onClick={() => setChatInput(q)}>
                                      <span className="followup-num">0{qi + 1}</span>
                                      <span className="followup-text">{q}</span>
                                    </div>
                                    <div className="followup-card-actions">
                                      <button
                                        type="button"
                                        className="followup-btn edit-btn"
                                        title="Edit question in message box"
                                        onClick={(e) => { e.stopPropagation(); setChatInput(q); }}
                                      >
                                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{width:12,height:12}}>
                                          <path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/>
                                        </svg>
                                        <span>Edit</span>
                                      </button>
                                      <button
                                        type="button"
                                        className="followup-btn send-now-btn"
                                        title="Send directly to Darukaa AI"
                                        disabled={isLoading || isTyping}
                                        onClick={(e) => { e.stopPropagation(); handleSendMessage(q); }}
                                      >
                                        <span>Ask AI</span>
                                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{width:11,height:11}}>
                                          <path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>
                                        </svg>
                                      </button>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                }

                if (msg.type === 'clarification') return (
                  <div key={msg.id} className="msg-row">
                    <div className="msg-avatar ai">🤔</div>
                    <div className="msg-body">
                      <div className="bubble-ai">
                        <div className="clarify-header">
                          <span style={{width:13,height:13,display:'inline-flex'}}><Icon.Help/></span>
                          More information needed
                        </div>
                        <div className="clarify-text">
                          {isLatest && isTyping
                            ? <TypewriterText text={msg.content} onFinished={() => setIsTyping(false)} />
                            : msg.content}
                        </div>
                        {msg.extracted_so_far && Object.values(msg.extracted_so_far).some(Boolean) && (
                          <div style={{marginBottom:10,display:'flex',flexWrap:'wrap',gap:5}}>
                            {msg.extracted_so_far.location && <span className="site-badge">📍 {msg.extracted_so_far.location}</span>}
                            {msg.extracted_so_far.land_use && <span className="site-badge">🌾 {msg.extracted_so_far.land_use}</span>}
                            {msg.extracted_so_far.water_info && <span className="site-badge">💧 {msg.extracted_so_far.water_info}</span>}
                            {msg.extracted_so_far.soil_info && <span className="site-badge">🪨 {msg.extracted_so_far.soil_info}</span>}
                            {msg.extracted_so_far.other_useful_features && <span className="site-badge">📋 {msg.extracted_so_far.other_useful_features}</span>}
                          </div>
                        )}
                        {msg.missing_variables?.length > 0 && (
                          <div className="chip-row">
                            <div className="chip-label">Click to add to your reply:</div>
                            {msg.missing_variables.map((v, i) => (
                              <button key={i} className="chip"
                                onClick={() => setChatInput(p => p ? `${p}, ${v}: ` : `${v}: `)}>
                                + {v}
                              </button>
                            ))}
                            <button className="chip" style={{borderColor:'rgba(251,191,36,0.4)',color:'#FBBF24'}}
                              onClick={() => setChatInput('Rainfall: 450mm, Crop: Wheat, Soil carbon: 0.7%')}>
                              🌾 Example values
                            </button>
                            <button className="chip" style={{borderColor:'rgba(52,211,153,0.4)',color:'var(--green)'}}
                              onClick={() => setActiveTab('scenario')}>
                              📊 Open Scenario Planner →
                            </button>
                          </div>
                        )}
                        {msg.timing && (
                          <div style={{marginTop:10, fontSize:11, color:'var(--text3)', fontFamily:'var(--mono)'}}>
                            ⚡ {msg.timing.total_time_s ?? 0.4}s
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );

                // Streaming message: render live token stream with top progress badge, or full pipeline stepper
                if (msg.type === 'streaming') {
                  if (msg.showText && msg.accumulated) {
                    return (
                      <div key={msg.id} className="msg-row">
                        <div className="msg-avatar ai">🌿</div>
                        <div className="msg-body" style={{flex:1, minWidth:0}}>
                          <div className="bubble-ai rec-bubble">
                            <div className="stream-pipeline-badge">
                              <div className="stream-badge-left">
                                <span className="stream-badge-pulse" />
                                <span className="stream-badge-title">
                                  {msg.stage >= 4 ? 'Validating & formatting science-backed plan…' : 'Synthesising multi-metric ecological reasoning…'}
                                </span>
                              </div>
                              <div className="stream-badge-stages">
                                {PIPELINE_STAGES.map((stage, si) => {
                                  const isDone = si < msg.stage;
                                  const isActive = si === msg.stage;
                                  return (
                                    <span
                                      key={stage.id}
                                      className={`stream-mini-stage ${isDone ? 'done' : isActive ? 'active' : 'pending'}`}
                                      title={stage.label}
                                    >
                                      {isDone ? '✓' : isActive ? <span className="mini-spinner" /> : stage.icon}
                                      <span className="mini-label">{stage.id}</span>
                                    </span>
                                  );
                                })}
                              </div>
                            </div>
                            <div className="chat-report">
                              <ParsedReport text={msg.accumulated} isDone={false} />
                            </div>
                            <span className="typing-cursor" />
                          </div>
                        </div>
                      </div>
                    );
                  }
                  return (
                    <div key={msg.id} className="msg-row">
                      <div className="msg-avatar ai">🌿</div>
                      <div className="msg-body">
                        <div className="pipeline-loader">
                          <div className="pipeline-header">
                            <span className="pipeline-title">AI Pipeline Running</span>
                          </div>
                          <div className="pipeline-steps">
                            {PIPELINE_STAGES.map((stage, si) => {
                              const isDone = si < msg.stage;
                              const isActive = si === msg.stage;
                              return (
                                <div key={stage.id} className={`pipeline-step${isDone ? ' done' : isActive ? ' active' : ' pending'}`}>
                                  <div className="pipeline-step-icon">
                                    {isDone ? '✓' : isActive ? <div className="step-spinner" /> : stage.icon}
                                  </div>
                                  <div className="pipeline-step-body">
                                    <div className="pipeline-step-label">{stage.label}</div>
                                    {isActive && <div className="pipeline-step-sub">{stage.sub}</div>}
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                          <div className="pipeline-tip">
                            <span>💡</span>
                            <span>Tip: Prefer structured fields? You can also use the <button type="button" className="pipeline-tip-link" onClick={() => setActiveTab('scenario')}>Scenario Planner</button> for custom numerical constraints.</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                }

                // Legacy recommendation fallback
                if (msg.recommendation) {
                  return (
                    <div key={msg.id} className="msg-row">
                      <div className="msg-avatar ai">🌿</div>
                      <div className="msg-body" style={{flex:1, minWidth:0}}>
                        <div className="bubble-ai rec-bubble">
                          <div className="rec-title-row">
                            <span className="rec-main-title">
                              <span style={{marginRight:8}}>🌿</span>{msg.title || 'Ecological Intervention Plan'}
                            </span>
                          </div>
                          <div className="report-divider" />
                          <RecommendationReport
                            msg={msg}
                            animate={isLatest && isTyping}
                            onFinished={() => setIsTyping(false)}
                          />
                        </div>
                      </div>
                    </div>
                  );
                }

                return null;
              })}

              {/* Error */}
              {chatError && (
                <div className="error-box">
                  <span style={{width:16,height:16,display:'inline-flex'}}><Icon.Alert/></span>
                  {chatError}
                </div>
              )}

              <div ref={chatBottomRef} />
            </div>

            {/* Input Bar */}
            <div className="chat-input-wrap">
              <form onSubmit={(e) => { e.preventDefault(); handleSendMessage(); }}>
                <div className="chat-input-inner">
                  <input
                    className="chat-input"
                    type="text"
                    value={chatInput}
                    onChange={e => setChatInput(e.target.value)}
                    disabled={isLoading || isTyping}
                    placeholder={
                      isTyping ? 'AI is responding…'
                        : isLoading ? 'Searching knowledge base…'
                        : 'Describe your land conditions, soil health, biodiversity challenges…'
                    }
                  />
                  <button type="submit" className="send-btn" disabled={!chatInput.trim() || isLoading || isTyping}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{width:15,height:15,color:'#052e16'}}>
                      <path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/>
                    </svg>
                  </button>
                </div>
                <div className="input-meta">
                  <span>Backed by Qdrant · BGE-Base-v1.5 · 768-dim vectors</span>
                  <button type="button" onClick={handleNewChat}>↺ New conversation</button>
                </div>
              </form>
            </div>
          </div>
        )}

        {/* ── TAB: SCENARIO PLANNER ──────────────────────────────────── */}
        {activeTab === 'scenario' && (
          <div className="pane">
            <div className="scenario-pane">
              <div className="pane-card">
                <div className="pane-title">
                  <span style={{width:18,height:18,display:'inline-flex',color:'var(--green)'}}><Icon.Sliders/></span>
                  Custom Scenario Planner
                </div>
                <div className="pane-sub">
                  Type any custom values in the fields below, or click a suggestion chip to populate. All fields accept free text — suggestions are just starting points.
                </div>

                <form onSubmit={handleScenarioSubmit}>
                  <div className="form-grid">
                    {/* Region */}
                    <div className="form-field">
                      <label>Ecological Region / Climate Zone <span style={{color:'var(--red)'}}>*</span></label>
                      <input
                        className="form-input"
                        type="text"
                        required
                        value={scenarioForm.region}
                        onChange={e => sf('region', e.target.value)}
                        placeholder="e.g. Semi-Arid, Mediterranean, Indo-Gangetic Plains"
                      />
                      <div className="sugg-chips">
                        {FORM_SUGGESTIONS.region?.map((s, i) => (
                          <button type="button" key={i}
                            className={`sugg-chip${scenarioForm.region === s ? ' sel' : ''}`}
                            onClick={() => sf('region', s)}>{s}</button>
                        ))}
                      </div>
                    </div>

                    {/* Crop */}
                    <div className="form-field">
                      <label>Current Crop / Vegetation <span style={{color:'var(--red)'}}>*</span></label>
                      <input
                        className="form-input"
                        type="text"
                        required
                        value={scenarioForm.crop}
                        onChange={e => sf('crop', e.target.value)}
                        placeholder="e.g. Wheat, Maize, Cotton, Grapevine"
                      />
                      <div className="sugg-chips">
                        {FORM_SUGGESTIONS.crop?.map((s, i) => (
                          <button type="button" key={i}
                            className={`sugg-chip${scenarioForm.crop === s ? ' sel' : ''}`}
                            onClick={() => sf('crop', s)}>{s}</button>
                        ))}
                      </div>
                    </div>

                    {/* Land Use */}
                    <div className="form-field">
                      <label>Land Use / Farming Practice <span style={{color:'var(--red)'}}>*</span></label>
                      <input
                        className="form-input"
                        type="text"
                        required
                        value={scenarioForm.land_use}
                        onChange={e => sf('land_use', e.target.value)}
                        placeholder="e.g. Monoculture, Agroforestry, Crop Rotation"
                      />
                      <div className="sugg-chips">
                        {FORM_SUGGESTIONS.land_use?.map((s, i) => (
                          <button type="button" key={i}
                            className={`sugg-chip${scenarioForm.land_use === s ? ' sel' : ''}`}
                            onClick={() => sf('land_use', s)}>{s}</button>
                        ))}
                      </div>
                    </div>

                    {/* Soil Organic Carbon (%) - strictly numeric */}
                    <div className="form-field">
                      <label>
                        Soil Organic Carbon (%) <span style={{color:'var(--red)'}}>*</span>
                        <span style={{fontSize:10.5,color:'var(--text3)',marginLeft:4}}>(digits only)</span>
                      </label>
                      <input
                        className="form-input"
                        type="text"
                        inputMode="decimal"
                        required
                        value={scenarioForm.soil_organic_carbon}
                        onChange={e => {
                          const val = e.target.value.replace(/[^0-9.]/g, '');
                          sf('soil_organic_carbon', val);
                        }}
                        placeholder="e.g. 0.3, 1.2"
                      />
                      <div className="sugg-chips">
                        {FORM_SUGGESTIONS.soil_organic_carbon?.map((s, i) => (
                          <button type="button" key={i}
                            className={`sugg-chip${scenarioForm.soil_organic_carbon === s ? ' sel' : ''}`}
                            onClick={() => sf('soil_organic_carbon', s)}>{s}%</button>
                        ))}
                      </div>
                    </div>

                    {/* Annual Rainfall (mm) - strictly integer */}
                    <div className="form-field">
                      <label>
                        Annual Rainfall (mm) <span style={{color:'var(--red)'}}>*</span>
                        <span style={{fontSize:10.5,color:'var(--text3)',marginLeft:4}}>(integers only)</span>
                      </label>
                      <input
                        className="form-input"
                        type="text"
                        inputMode="numeric"
                        required
                        value={scenarioForm.rainfall}
                        onChange={e => {
                          const val = e.target.value.replace(/[^0-9]/g, '');
                          sf('rainfall', val);
                        }}
                        placeholder="e.g. 450, 980"
                      />
                      <div className="sugg-chips">
                        {FORM_SUGGESTIONS.rainfall?.map((s, i) => (
                          <button type="button" key={i}
                            className={`sugg-chip${scenarioForm.rainfall === s ? ' sel' : ''}`}
                            onClick={() => sf('rainfall', s)}>{s} mm</button>
                        ))}
                      </div>
                    </div>

                    {/* Soil Moisture & Texture */}
                    <div className="form-field">
                      <label>Soil Moisture & Texture <span style={{color:'var(--red)'}}>*</span></label>
                      <input
                        className="form-input"
                        type="text"
                        required
                        value={scenarioForm.soil_moisture}
                        onChange={e => sf('soil_moisture', e.target.value)}
                        placeholder="e.g. Sandy Loam, Clay Loam, Compacted"
                      />
                      <div className="sugg-chips">
                        {FORM_SUGGESTIONS.soil_moisture?.map((s, i) => (
                          <button type="button" key={i}
                            className={`sugg-chip${scenarioForm.soil_moisture === s ? ' sel' : ''}`}
                            onClick={() => sf('soil_moisture', s)}>{s}</button>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Habitat Diversity (full width) */}
                  <div className="form-field" style={{marginBottom:16}}>
                    <label>Habitat Diversity Status <span style={{color:'var(--red)'}}>*</span></label>
                    <input
                      className="form-input"
                      type="text"
                      required
                      value={scenarioForm.habitat_diversity}
                      onChange={e => sf('habitat_diversity', e.target.value)}
                      placeholder="e.g. Severe Decline, Moderate / Patchy"
                    />
                    <div className="sugg-chips">
                      {FORM_SUGGESTIONS.habitat_diversity.map((s, i) => (
                        <button type="button" key={i}
                          className={`sugg-chip${scenarioForm.habitat_diversity === s ? ' sel' : ''}`}
                          onClick={() => sf('habitat_diversity', s)}>{s}</button>
                      ))}
                    </div>
                  </div>

                  {/* Other Useful Information / Field Observations (OPTIONAL) */}
                  <div className="form-field" style={{marginBottom:20}}>
                    <label>
                      Other Useful Information / Specific Observations
                      <span style={{fontSize:11,color:'var(--text3)',marginLeft:6,fontWeight:400}}>(Optional)</span>
                    </label>
                    <input
                      className="form-input"
                      type="text"
                      value={scenarioForm.other_useful_info || ''}
                      onChange={e => sf('other_useful_info', e.target.value)}
                      placeholder="e.g. Soil crusting after rain, high pest pressure, past chemical fertilizers, steep slope, depleted earthworms..."
                    />
                    <div className="sugg-chips">
                      {FORM_SUGGESTIONS.other_useful_info?.map((s, i) => (
                        <button type="button" key={i}
                          className={`sugg-chip${scenarioForm.other_useful_info === s ? ' sel' : ''}`}
                          onClick={() => sf('other_useful_info', s)}>{s}</button>
                      ))}
                    </div>
                  </div>

                  <div className="form-footer">
                    <span className="form-hint"><span style={{color:'var(--red)'}}>*</span> Mandatory fields · Optional observations enrich multi-variable synthesis</span>
                    <button type="submit" className="submit-btn" disabled={scenarioLoading || isLoading}>
                      {scenarioLoading || isLoading
                        ? <><div className="spinner" /> Launching pipeline…</>
                        : <><span style={{width:14,height:14,display:'inline-flex'}}><Icon.Sparkles/></span> Launch AI Pipeline in Chat</>}
                    </button>
                  </div>
                </form>
              </div>

              {scenarioError && (
                <div className="error-box" style={{marginBottom:16}}><span style={{width:16,height:16,display:'inline-flex'}}><Icon.Alert/></span> {scenarioError}</div>
              )}

              {scenarioResult?.recommendation && (
                <div className="result-card">
                  <div className="pane-title" style={{marginBottom:12}}>
                    <span style={{width:18,height:18,display:'inline-flex',color:'var(--green)'}}><Icon.Shield/></span>
                    {scenarioResult.recommendation.title}
                  </div>
                  {scenarioResult.timing && (
                    <div className="timing-row" style={{marginBottom:14}}>
                      <span className="timing-pill green">⚡ {scenarioResult.timing.total_time_s}s total</span>
                      <span className="timing-pill cyan">retrieval {scenarioResult.timing.retrieval_time_s}s</span>
                      <span className="timing-pill purple">reasoning {scenarioResult.timing.thinking_time_s}s</span>
                    </div>
                  )}
                  <div className="strategy-box">
                    <div className="strategy-label">Primary Strategy</div>
                    <div className="strategy-text">{scenarioResult.recommendation.primary_intervention}</div>
                  </div>
                  {scenarioResult.recommendation.environmental_assessment && (
                    <div className="assessment-box">{scenarioResult.recommendation.environmental_assessment}</div>
                  )}
                  {scenarioResult.recommendation.recommendations?.length > 0 && (
                    <div className="rec-items">
                      {scenarioResult.recommendation.recommendations.map((rec, ri) => (
                        <div key={ri} className="rec-item">
                          <div className="rec-item-title">
                            <span>📌 {rec.what_to_do}</span>
                            {rec.time_horizon && <span className="time-tag">{rec.time_horizon}</span>}
                          </div>
                          <div className="rec-why"><strong style={{color:'var(--text2)'}}>Why: </strong>{rec.why_it_works}</div>
                          {rec.measurable_outcome && (
                            <div className="rec-impact"><strong>Impact: </strong>{rec.measurable_outcome}</div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── TAB: EVIDENCE EXPLORER ─────────────────────────────────── */}
        {activeTab === 'explorer' && (
          <div className="pane">
            <div className="explorer-pane">
              <div className="pane-card" style={{marginBottom:16}}>
                <div className="pane-title" style={{marginBottom:4}}>
                  <span style={{width:18,height:18,display:'inline-flex',color:'var(--cyan)'}}><Icon.Database/></span>
                  Evidence Knowledge Base
                </div>
                <div className="pane-sub">
                  Search across 7,978 verified scientific passages from FAO, IPCC, IPBES, and GWO reports.
                </div>

                {/* Search Mode Switcher (2 Options: Language vs ID) */}
                <div className="search-mode-tabs">
                  <button
                    type="button"
                    className={`search-mode-btn ${searchMode === 'semantic' ? 'active' : ''}`}
                    onClick={() => {
                      setSearchMode('semantic');
                      setSearchResults(null);
                      setSearchQuery('cover crops soil organic carbon water retention');
                    }}
                  >
                    🧠 Semantic Search (Language)
                  </button>
                  <button
                    type="button"
                    className={`search-mode-btn ${searchMode === 'id' ? 'active' : ''}`}
                    onClick={() => {
                      setSearchMode('id');
                      setSearchResults(null);
                      setSearchQuery('E5774');
                    }}
                  >
                    🔢 Evidence ID Search (e.g. E5774)
                  </button>
                </div>

                <form onSubmit={handleExplorerSearch}>
                  <div className="search-bar">
                    <input
                      className="search-input"
                      type="text"
                      value={searchQuery}
                      onChange={e => setSearchQuery(e.target.value)}
                      placeholder={
                        searchMode === 'id'
                          ? 'Enter evidence ID or number (e.g. E5774, E0181, 5774)...'
                          : 'Describe ecological practice, crop, or soil condition (e.g. cover crops for soil moisture)...'
                      }
                    />
                    <button type="submit" className="search-btn" disabled={isSearching}>
                      <span style={{width:14,height:14,display:'inline-flex'}}><Icon.Search/></span>
                      {isSearching ? 'Searching…' : 'Search'}
                    </button>
                  </div>
                </form>

                {/* Quick test suggestion chips */}
                <div style={{marginTop:10,display:'flex',gap:6,flexWrap:'wrap',alignItems:'center'}}>
                  <span style={{fontSize:11,color:'var(--text3)'}}>Quick test:</span>
                  {searchMode === 'id' ? (
                    ['E5774', 'E0181', 'E0905', 'E2277', 'E3117'].map((id) => (
                      <button
                        key={id}
                        type="button"
                        className="sugg-chip"
                        style={{padding:'2px 8px',fontSize:11}}
                        onClick={() => {
                          setSearchQuery(id);
                          handleExplorerSearch(null, id, 'id');
                        }}
                      >
                        {id}
                      </button>
                    ))
                  ) : (
                    [
                      'cover crops soil organic carbon',
                      'pollinator biodiversity hedgerows',
                      'semi-arid soil moisture retention',
                      'legume rotation microbial diversity'
                    ].map((term) => (
                      <button
                        key={term}
                        type="button"
                        className="sugg-chip"
                        style={{padding:'2px 8px',fontSize:11}}
                        onClick={() => {
                          setSearchQuery(term);
                          handleExplorerSearch(null, term, 'semantic');
                        }}
                      >
                        {term}
                      </button>
                    ))
                  )}
                </div>
              </div>

              {/* Search Results */}
              {searchResults?.chunks?.length > 0 && (
                <div>
                  <div style={{fontSize:12,color:'var(--text3)',marginBottom:12,fontFamily:'var(--mono)',display:'flex',justifyContent:'space-between',alignItems:'center'}}>
                    <span>
                      Found <strong style={{color:'var(--green)'}}>{searchResults.total_found}</strong> relevant passages {searchMode === 'id' ? 'matching ID' : 'with high semantic similarity'}
                    </span>
                    <span>
                      Showing top {Math.min(visibleCount, searchResults.chunks.length)} of {searchResults.chunks.length}
                    </span>
                  </div>

                  {searchResults.chunks.slice(0, visibleCount).map((ev, i) => (
                    <div key={i} className="ev-card">
                      <div className="ev-card-meta">
                        <span className="ev-id">[{ev.evidence_id || `EV${String(i+1).padStart(3,'0')}`}]</span>
                        <span className="ev-doc">{ev.doc_title}</span>
                        <span className="ev-score" style={{color: (ev.score || 0) >= 0.7 ? 'var(--green)' : 'var(--cyan)'}}>
                          {searchMode === 'id' ? 'exact match' : `score ${(ev.score || 0).toFixed(3)}`}
                        </span>
                      </div>
                      {ev.section && <div className="ev-section">{ev.section}</div>}
                      <div className="ev-card-text">"{ev.text}"</div>
                    </div>
                  ))}

                  {visibleCount < searchResults.chunks.length && (
                    <div style={{textAlign:'center',marginTop:16,marginBottom:24}}>
                      <button
                        type="button"
                        className="load-more-btn"
                        onClick={() => setVisibleCount(p => p + 5)}
                      >
                        Show More (+5 passages) · {searchResults.chunks.length - visibleCount} remaining
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* No results card */}
              {searchResults && !searchResults.chunks?.length && (
                <div className="no-results-card">
                  <div style={{fontSize:28,marginBottom:8}}>🔍</div>
                  <div style={{fontSize:15,fontWeight:600,color:'var(--text)',marginBottom:4}}>No matching passages found</div>
                  <div style={{fontSize:12.5,color:'var(--text3)',maxWidth:460,margin:'0 auto',lineHeight:1.5}}>
                    {searchMode === 'id'
                      ? `No evidence passage matches ID "${searchResults.query}". Valid IDs range from E0001 to E7978 (e.g. E5774).`
                      : `No passages met the semantic relevance threshold for "${searchResults.query}". Random zero-relevance passages have been filtered out. Try broader environmental terms or switch to Evidence ID Search.`}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
