import React, { useState, useEffect, useRef } from 'react';
import { 
  BarChart, Bar, Cell, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, CartesianGrid 
} from 'recharts';
import { 
  Activity, ShieldAlert, Sparkles, Send, DollarSign, BookOpen, 
  AlertCircle, FileText, CheckCircle2, TrendingDown, User, Heart, Calculator,
  TrendingUp, RefreshCw, Cpu, UserCheck, HelpCircle, Upload, FileCheck
} from 'lucide-react';
import './App.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Database columns map to human-readable names
const COLUMN_MAPPER = {
  // External Credit Scores
  'EXT_SOURCE_1': 'External Credit Score (Source 1)',
  'EXT_SOURCE_2': 'External Credit Score (Source 2)',
  'EXT_SOURCE_3': 'External Credit Score (Source 3)',
  
  // Financial Amounts
  'AMT_INCOME_TOTAL': 'Annual Income',
  'AMT_CREDIT': 'Requested Loan Credit',
  'AMT_ANNUITY': 'Loan Annuity (Monthly Payment)',
  'AMT_GOODS_PRICE': 'Goods Purchase Price',
  
  // Demographics / Profile
  'DAYS_BIRTH': 'Age',
  'DAYS_EMPLOYED': 'Employment Tenure',
  'DAYS_REGISTRATION': 'Address Registration Age',
  'DAYS_ID_PUBLISH': 'Identity Document Age',
  'OWN_CAR_AGE': 'Car Age',
  'CNT_CHILDREN': 'Number of Children',
  'CNT_FAM_MEMBERS': 'Total Family Members',
  
  // Flag indicators
  'FLAG_OWN_REALTY': 'Real Estate Property Ownership',
  'FLAG_OWN_CAR': 'Vehicle / Car Ownership',
  'CODE_GENDER': 'Applicant Gender',
  'NAME_CONTRACT_TYPE': 'Type of Loan Credit',
  'NAME_TYPE_SUITE': 'Loan Application Companion',
  'NAME_INCOME_TYPE': 'Employment / Income Type',
  'NAME_EDUCATION_TYPE': 'Highest Education Level',
  'NAME_FAMILY_STATUS': 'Marital / Family Status',
  'NAME_HOUSING_TYPE': 'Residential Housing Status',
  
  // Occupation / Org
  'OCCUPATION_TYPE': 'Professional Occupation',
  'ORGANIZATION_TYPE': 'Employer Business Sector',
  
  // Address flags
  'REG_REGION_NOT_LIVE_REGION': 'Region Address Mismatch Check',
  'REG_REGION_NOT_WORK_REGION': 'Work Region Address Mismatch Check',
  'LIVE_REGION_NOT_WORK_REGION': 'Live vs Work Region Mismatch Check',
  'REG_CITY_NOT_LIVE_CITY': 'City Address Mismatch Check',
  'REG_CITY_NOT_WORK_CITY': 'Work City Address Mismatch Check',
  'LIVE_CITY_NOT_WORK_CITY': 'Live vs Work City Mismatch Check',
  
  // Document counts / flags
  'FLAG_DOCUMENT_3': 'Main Document Verification',
  'FLAG_DOCUMENT_2': 'Supplementary Document A',
  'FLAG_DOCUMENT_4': 'Supplementary Document B',
  
  // Bureau requests
  'AMT_REQ_CREDIT_BUREAU_HOUR': 'Bureau Inquiry Frequency (Hour)',
  'AMT_REQ_CREDIT_BUREAU_DAY': 'Bureau Inquiry Frequency (Day)',
  'AMT_REQ_CREDIT_BUREAU_WEEK': 'Bureau Inquiry Frequency (Week)',
  'AMT_REQ_CREDIT_BUREAU_MON': 'Bureau Inquiry Frequency (Month)',
  'AMT_REQ_CREDIT_BUREAU_QRT': 'Bureau Inquiry Frequency (Quarter)',
  'AMT_REQ_CREDIT_BUREAU_YEAR': 'Bureau Inquiry Frequency (Year)',
  
  // Social Circle / Default Counts
  'DEF_30_CNT_SOCIAL_CIRCLE': 'Social Defaults (30 Days)',
  'DEF_60_CNT_SOCIAL_CIRCLE': 'Social Defaults (60 Days)',
  'OBS_30_CNT_SOCIAL_CIRCLE': 'Social Inquiries (30 Days)',
  'OBS_60_CNT_SOCIAL_CIRCLE': 'Social Inquiries (60 Days)',
  
  // Real Estate features
  'LIVINGAREA_MEDI': 'Median Apartment Living Area',
  'YEARS_BUILD_MODE': 'Building Construction Age (Mode)',
  'FONDKAPREMONT_MODE': 'Building Manager Standard',
  'FONDKAPREMONT_MODE_reg_oper_account': 'Housing Management Operator',
  
  // Other flags
  'FLAG_PHONE': 'Home Telephone Supplied',
  'EXT_SOURCE_MEAN': 'Average External Bureau Rating',
};

const getReadableName = (colName) => {
  return COLUMN_MAPPER[colName] || colName.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
};

const formatValue = (key, val) => {
  if (val === null || val === undefined) return 'N/A';
  
  // Format age from negative days to positive years
  if (key === 'DAYS_BIRTH') {
    const age = Math.abs(val) / 365.25;
    return `${Math.floor(age)} Years`;
  }
  
  // Format employment tenure from negative days
  if (key === 'DAYS_EMPLOYED') {
    if (val > 0) return 'Unemployed / Retired';
    const years = Math.abs(val) / 365.25;
    if (years < 1) {
      return `${Math.floor(Math.abs(val) / 30.4)} Months`;
    }
    return `${Math.floor(years)} Years`;
  }
  
  if (['DAYS_REGISTRATION', 'DAYS_ID_PUBLISH'].includes(key)) {
    const years = Math.abs(val) / 365.25;
    return `${Math.floor(years)} Years Ago`;
  }
  
  // Format currencies
  if (['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE'].includes(key)) {
    return `Rs. ${Math.round(val).toLocaleString()}`;
  }

  // Format 1/0 flags for mismatched addresses
  if (key.includes('_NOT_')) {
    return val === 1 || val === '1' ? '⚠️ Mismatch Found' : '✅ Verified Match';
  }

  // Format general 1/0 flags as Yes/No
  if (key.startsWith('FLAG_DOCUMENT_') || key.startsWith('FLAG_OWN_')) {
    return val === 1 || val === '1' || val === 'Y' ? 'Yes' : 'No';
  }

  // Format Genders
  if (key === 'CODE_GENDER') {
    if (val === 'M') return 'Male';
    if (val === 'F') return 'Female';
    return val;
  }

  // Format regional risk ratings
  if (key.startsWith('REGION_RATING_')) {
    const rating = parseInt(val);
    if (rating === 1) return 'Tier 1 (Lowest Risk)';
    if (rating === 2) return 'Tier 2 (Moderate Risk)';
    if (rating === 3) return 'Tier 3 (Highest Risk)';
    return val;
  }

  // Format process start hour
  if (key === 'HOUR_APPR_PROCESS_START') {
    const h = parseInt(val);
    const ampm = h >= 12 ? 'PM' : 'AM';
    const displayHour = h % 12 === 0 ? 12 : h % 12;
    return `${displayHour}:00 ${ampm}`;
  }
  
  return val.toString();
};

// Interactive Neural Synapse Logo
const NeuroFinanceLogo = () => {
  const [hovered, setHovered] = useState(false);
  
  return (
    <div 
      className="logo-wrapper"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{ display: 'flex', alignItems: 'center', gap: '14px', cursor: 'pointer' }}
    >
      <svg width="40" height="40" viewBox="0 0 100 100" style={{ transition: 'transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275)', transform: hovered ? 'scale(1.08) rotate(3deg)' : 'scale(1)' }}>
        {/* Background Circle */}
        <circle cx="50" cy="50" r="46" fill="url(#logoBgGrad)" stroke="var(--accent-purple)" strokeWidth="1.5" style={{ opacity: 0.8 }} />
        
        {/* Neural Network Nodes and Connections */}
        <line x1="30" y1="50" x2="50" y2="30" stroke="rgba(255,255,255,0.2)" strokeWidth="1.5" className={hovered ? "synapse-line-fast" : ""} />
        <line x1="30" y1="50" x2="50" y2="70" stroke="rgba(255,255,255,0.2)" strokeWidth="1.5" className={hovered ? "synapse-line-fast" : ""} />
        <line x1="50" y1="30" x2="70" y2="50" stroke="var(--accent-purple)" strokeWidth="2" strokeDasharray="4 2" className={hovered ? "synapse-line-fast" : ""} />
        <line x1="50" y1="70" x2="70" y2="50" stroke="var(--accent-blue)" strokeWidth="2" strokeDasharray="4 2" className={hovered ? "synapse-line-fast" : ""} />
        <line x1="50" y1="30" x2="50" y2="70" stroke="rgba(255,255,255,0.15)" strokeWidth="1" />
        <line x1="30" y1="50" x2="70" y2="50" stroke="var(--accent-purple)" strokeWidth="1.5" className={hovered ? "synapse-line-fast" : ""} />
        
        {/* Financial Upward trend line */}
        <path d="M 25 65 Q 45 65 50 45 T 75 28" fill="none" stroke="var(--color-success)" strokeWidth="3" strokeLinecap="round" style={{ filter: 'drop-shadow(0 0 6px var(--color-success))', transition: 'stroke-dashoffset 2s', strokeDasharray: '100', strokeDashoffset: hovered ? '0' : '20' }} />
        
        {/* Synapse Nodes */}
        <circle cx="30" cy="50" r={hovered ? 6 : 5} fill="#fff" style={{ filter: 'drop-shadow(0 0 4px #fff)', transition: 'all 0.2s' }} />
        <circle cx="50" cy="30" r={hovered ? 5.5 : 4.5} fill="var(--accent-purple)" style={{ filter: 'drop-shadow(0 0 6px var(--accent-purple))', transition: 'all 0.2s' }} />
        <circle cx="50" cy="70" r={hovered ? 5.5 : 4.5} fill="var(--accent-blue)" style={{ filter: 'drop-shadow(0 0 6px var(--accent-blue))', transition: 'all 0.2s' }} />
        <circle cx="70" cy="50" r={hovered ? 6 : 5} fill="var(--color-success)" style={{ filter: 'drop-shadow(0 0 8px var(--color-success))', transition: 'all 0.2s' }} />
        
        <defs>
          <radialGradient id="logoBgGrad" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="var(--bg-secondary)" />
            <stop offset="100%" stopColor="#0c0a21" />
          </radialGradient>
        </defs>
      </svg>
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        <h1 style={{ 
          fontFamily: 'Space Grotesk, sans-serif', 
          fontSize: '22px', 
          fontWeight: 700, 
          margin: 0, 
          letterSpacing: '-0.5px',
          color: 'var(--text-primary)',
          textShadow: hovered ? '0 0 15px rgba(192, 132, 252, 0.65)' : 'none',
          transition: 'all 0.3s'
        }}>
          NeuroFinance.AI
        </h1>
        <span style={{ fontSize: '9px', fontWeight: 600, color: 'var(--text-secondary)', letterSpacing: '1px', textTransform: 'uppercase', marginTop: '2px' }}>
          Neural Risk & Decision Engine
        </span>
      </div>
    </div>
  );
};

// Floating Robot Mascot Component
const RobotMascot = () => (
  <svg width="70" height="70" viewBox="0 0 100 100" className="robot-mascot">
    {/* Antennas */}
    <line x1="50" y1="30" x2="50" y2="10" stroke="var(--accent-purple)" strokeWidth="3" strokeLinecap="round" />
    <circle cx="50" cy="8" r="5" fill="var(--accent-purple)" style={{ filter: 'drop-shadow(0 0 6px var(--accent-purple))' }} />
    <line x1="35" y1="30" x2="25" y2="15" stroke="var(--accent-blue)" strokeWidth="2.5" strokeLinecap="round" />
    <circle cx="23" cy="13" r="4" fill="var(--accent-blue)" style={{ filter: 'drop-shadow(0 0 4px var(--accent-blue))' }} />
    <line x1="65" y1="30" x2="75" y2="15" stroke="var(--accent-blue)" strokeWidth="2.5" strokeLinecap="round" />
    <circle cx="77" cy="13" r="4" fill="var(--accent-blue)" style={{ filter: 'drop-shadow(0 0 4px var(--accent-blue))' }} />
    
    {/* Neck */}
    <rect x="42" y="70" width="16" height="8" rx="3" fill="#1f2937" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
    
    {/* Head */}
    <rect x="20" y="24" width="60" height="48" rx="14" fill="#0d0c22" stroke="var(--accent-purple)" strokeWidth="2" style={{ filter: 'drop-shadow(0 0 10px rgba(192, 132, 252, 0.35))' }} />
    
    {/* Ears */}
    <rect x="14" y="38" width="6" height="18" rx="2" fill="var(--accent-blue)" />
    <rect x="80" y="38" width="6" height="18" rx="2" fill="var(--accent-blue)" />
    
    {/* Screen Face */}
    <rect x="26" y="30" width="48" height="34" rx="8" fill="#050508" stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
    
    {/* Eyes */}
    <circle cx="38" cy="44" r="5" fill="var(--color-success)" className="robot-eye-glow" style={{ filter: 'drop-shadow(0 0 8px var(--color-success))' }} />
    <circle cx="62" cy="44" r="5" fill="var(--color-success)" className="robot-eye-glow" style={{ filter: 'drop-shadow(0 0 8px var(--color-success))' }} />
    
    {/* Mouth */}
    <path d="M 40 54 Q 50 59 60 54" stroke="var(--accent-blue)" strokeWidth="2.5" strokeLinecap="round" fill="none" style={{ filter: 'drop-shadow(0 0 4px var(--accent-blue))' }} />
  </svg>
);

// Simple Markdown-to-HTML parser helper for chatbot bubble formatting
const renderFormattedText = (text) => {
  if (!text) return null;
  
  return text.split('\n').map((line, lineIdx) => {
    let content = line.trim();
    if (!content) return <div key={lineIdx} style={{ height: '8px' }} />;

    // Headers
    if (content.startsWith('### ')) {
      return <h4 key={lineIdx} style={{ margin: '14px 0 6px 0', fontSize: '15px', fontWeight: 600, color: 'var(--accent-purple)' }}>{content.replace('### ', '')}</h4>;
    }
    if (content.startsWith('#### ')) {
      return <h5 key={lineIdx} style={{ margin: '10px 0 4px 0', fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>{content.replace('#### ', '')}</h5>;
    }

    // Bullet points
    if (content.startsWith('- ') || content.startsWith('* ')) {
      const cleanText = content.replace(/^[-*]\s+/, '');
      return (
        <li key={lineIdx} style={{ marginLeft: '12px', listStyleType: 'disc', marginBottom: '4px' }}>
          {parseBoldText(cleanText)}
        </li>
      );
    }

    // Numbered list
    const numListMatch = content.match(/^(\d+)\.\s+(.*)/);
    if (numListMatch) {
      return (
        <li key={lineIdx} style={{ marginLeft: '12px', listStyleType: 'decimal', marginBottom: '4px' }}>
          {parseBoldText(numListMatch[2])}
        </li>
      );
    }

    return <p key={lineIdx} style={{ margin: '0 0 6px 0' }}>{parseBoldText(content)}</p>;
  });
};

const parseBoldText = (text) => {
  const parts = text.split(/\*\*([^*]+)\*\*/g);
  return parts.map((part, index) => {
    // Odd indices contain bold text
    if (index % 2 === 1) {
      return <strong key={index} style={{ fontWeight: 600, color: '#fff' }}>{part}</strong>;
    }
    // Handle inline code quotes
    const codeParts = part.split(/`([^`]+)`/g);
    return codeParts.map((subPart, subIndex) => {
      if (subIndex % 2 === 1) {
        return <code key={subIndex} style={{ background: 'rgba(255,255,255,0.06)', padding: '2px 4px', borderRadius: '4px', fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--accent-purple)' }}>{subPart}</code>;
      }
      return subPart;
    });
  });
};

function App() {
  const [theme, setTheme] = useState('dark');

  useEffect(() => {
    document.body.setAttribute('data-theme', theme);
  }, [theme]);

  const [customers, setCustomers] = useState([]);
  const [selectedId, setSelectedId] = useState('');
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('risk');
  const [backendStatus, setBackendStatus] = useState('offline');

  // What-If Simulation State
  const [simulating, setSimulating] = useState(false);
  const [simulatedResult, setSimulatedResult] = useState(null);
  const [simInputs, setSimInputs] = useState({
    income: 150000,
    credit: 500000,
    goods: 450000,
    extSource: 0.5
  });

  // Document Intelligence State
  const [uploading, setUploading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [analyzingDoc, setAnalyzingDoc] = useState(false);

  // NeuroBot Chat State
  const [messages, setMessages] = useState([
    { 
      sender: 'bot', 
      text: "### Welcome to NeuroBot Decision Assistant!\nI can perform real-time credit checks, run What-If simulations, calculate repayment EMIs, or lookup lending policies inside our financial knowledge base.\n\n* **Analyze risk**: Ask me to assess a client ID (e.g. `Assess client 100002`)\n* **Policy check**: Ask about address verification or KYC circulars\n* **Calculator**: Ask to calculate EMIs (e.g. `Calculate EMI for 12 lakh at 8% for 5 years`)\n\nWhat scenario shall we evaluate?" 
    }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef(null);

  // EMI State
  const [emiPrincipal, setEmiPrincipal] = useState(800000);
  const [emiRate, setEmiRate] = useState(9.0);
  const [emiMonths, setEmiMonths] = useState(60);
  const [emiResult, setEmiResult] = useState(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Initial loads
  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then(res => res.json())
      .then(data => {
        if (data.status === 'ok') setBackendStatus('online');
      })
      .catch(() => setBackendStatus('offline'));

    setLoading(true);
    fetch(`${API_URL}/api/prediction/samples`)
      .then(res => {
        if (!res.ok) throw new Error("API backend loading or training model...");
        return res.json();
      })
      .then(data => {
        setCustomers(data);
        if (data.length > 0) {
          setSelectedId(data[0].sk_id_curr.toString());
        }
        setLoading(false)
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  // Update simulator sliders when report changes
  useEffect(() => {
    if (!report) return;
    
    // Convert baseline values from loaded customer
    const income = report.profile.AMT_INCOME_TOTAL || 150000;
    const credit = report.profile.AMT_CREDIT || 500000;
    const goods = report.profile.AMT_GOODS_PRICE || 450000;
    
    // Average external source score
    const ext1 = report.profile.EXT_SOURCE_1 || 0.5;
    const ext2 = report.profile.EXT_SOURCE_2 || 0.5;
    const ext3 = report.profile.EXT_SOURCE_3 || 0.5;
    const extSource = (ext1 + ext2 + ext3) / 3;

    setSimInputs({ income, credit, goods, extSource });
    setSimulatedResult(null); // Reset simulation
  }, [report]);

  // Load customer
  useEffect(() => {
    if (!selectedId) return;
    setLoading(true);
    setError(null);
    
    fetch(`${API_URL}/api/prediction/customer/${selectedId}`)
      .then(res => {
        if (!res.ok) throw new Error("Failed to load customer prediction details.");
        return res.json();
      })
      .then(data => {
        setReport(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [selectedId]);

  // Chat message submit
  const handleSendMessage = async (textToSend) => {
    const text = textToSend || chatInput;
    if (!text.trim()) return;

    setMessages(prev => [...prev, { sender: 'user', text }]);
    if (!textToSend) setChatInput('');
    setChatLoading(true);

    try {
      const response = await fetch(`${API_URL}/api/neurobot/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: text,
          customer_id: selectedId ? parseInt(selectedId) : 100002
        })
      });
      const data = await response.json();
      setMessages(prev => [...prev, { sender: 'bot', text: data.answer }]);
    } catch (err) {
      setMessages(prev => [...prev, { sender: 'bot', text: "Error communicating with NeuroBot. Please verify backend is running." }]);
    } finally {
      setChatLoading(false);
    }
  };

  // Run What-If Simulation
  const handleRunSimulation = async () => {
    if (!report) return;
    setSimulating(true);

    // Create a modified copy of the current profile
    const simulatedProfile = { ...report.profile };
    simulatedProfile.AMT_INCOME_TOTAL = parseFloat(simInputs.income);
    simulatedProfile.AMT_CREDIT = parseFloat(simInputs.credit);
    simulatedProfile.AMT_GOODS_PRICE = parseFloat(simInputs.goods);
    
    // Map simulated external rating back to the three source fields
    simulatedProfile.EXT_SOURCE_1 = parseFloat(simInputs.extSource);
    simulatedProfile.EXT_SOURCE_2 = parseFloat(simInputs.extSource);
    simulatedProfile.EXT_SOURCE_3 = parseFloat(simInputs.extSource);

    try {
      const response = await fetch(`${API_URL}/api/prediction/custom`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: simulatedProfile })
      });
      const data = await response.json();
      setSimulatedResult(data);
    } catch (err) {
      alert("Error executing custom simulation: " + err.message);
    } finally {
      setSimulating(false);
    }
  };

  // Document upload handler
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setUploadedFile(null);

    const formData = new FormData();
    formData.append("file", file);

    fetch(`${API_URL}/api/documents/classify`, {
      method: "POST",
      body: formData
    })
      .then(res => {
        if (!res.ok) throw new Error("CNN document classification failed.");
        return res.json();
      })
      .then(data => {
        setUploadedFile(data);
        setUploading(false);
      })
      .catch(err => {
        alert(err.message);
        setUploading(false);
      });
  };

  // Run end-to-end document risk integration
  const handleAnalyzeDocRisk = () => {
    if (!uploadedFile || !selectedId) return;
    setAnalyzingDoc(true);

    fetch(`${API_URL}/api/documents/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        customer_id: parseInt(selectedId),
        filename: uploadedFile.filename
      })
    })
      .then(res => {
        if (!res.ok) throw new Error("Document risk simulation failed.");
        return res.json();
      })
      .then(data => {
        setSimulatedResult(data.risk_report);
        setAnalyzingDoc(false);
        alert("Verification merged! Calculated default metrics have been updated using the extracted document values.");
      })
      .catch(err => {
        alert(err.message);
        setAnalyzingDoc(false);
      });
  };

  // EMI Calculator Trigger
  const handleCalculateEMI = () => {
    fetch(`${API_URL}/api/tools/emi`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        principal: parseFloat(emiPrincipal),
        annual_rate: parseFloat(emiRate),
        months: parseInt(emiMonths)
      })
    })
      .then(res => res.json())
      .then(data => setEmiResult(data))
      .catch(() => {
        // Fallback calculations if backend unavailable
        const r = emiRate / 12 / 100;
        const n = emiMonths;
        const p = emiPrincipal;
        const emi = (p * r * Math.pow(1 + r, n)) / (Math.pow(1 + r, n) - 1);
        const totalPayment = emi * n;
        const totalInterest = totalPayment - p;
        setEmiResult({
          emi: parseFloat(emi.toFixed(2)),
          total_payment: parseFloat(totalPayment.toFixed(2)),
          total_interest: parseFloat(totalInterest.toFixed(2))
        });
      });
  };

  const prefillAffordableEMI = () => {
    if (!report) return;
    const monthlyIncome = (report.profile.AMT_INCOME_TOTAL || 150000) / 12;
    // Target 30% DTI (safe / fully affordable zone under 40%)
    const targetEMI = monthlyIncome * 0.30;
    
    // Solve for principal: P = (EMI * ((1+r)^n - 1)) / (r * (1+r)^n)
    const r = emiRate / 12 / 100;
    const n = emiMonths;
    const p = (targetEMI * (Math.pow(1 + r, n) - 1)) / (r * Math.pow(1 + r, n));
    
    setEmiPrincipal(Math.round(p));
    // Immediately calculate EMI
    setTimeout(() => {
      handleCalculateEMI();
    }, 50);
  };

  useEffect(() => {
    handleCalculateEMI();
  }, []);

  // Compute DTI Affordability Status
  const getDTIDetails = () => {
    if (!report || !emiResult) return null;
    const monthlyIncome = (report.profile.AMT_INCOME_TOTAL || 150000) / 12;
    const emi = emiResult.emi;
    const dti = (emi / monthlyIncome) * 100;

    let status = 'affordable';
    let badgeText = 'Affordable';
    let advisory = 'This loan is fully affordable. The monthly installment fits safely within standard banking limits (under 40% of income).';

    if (dti > 60) {
      status = 'critical';
      badgeText = 'Critical / Stressed';
      advisory = 'Warning: This repayment plan is highly critical, consuming over 60% of monthly income. Default risk is extremely high. Consider lowering principal or increasing tenure.';
    } else if (dti > 40) {
      status = 'stretched';
      badgeText = 'Stretched';
      advisory = 'Repayment is stretched (40% - 60% of income). Approvals will require additional collateral or guarantees. Restructuring tenure is recommended to bring DTI under 40%.';
    }

    return {
      dti: Math.round(dti),
      status,
      badgeText,
      advisory,
      monthlyIncome: Math.round(monthlyIncome)
    };
  };

  const dtiDetails = getDTIDetails();

  // Recharts SHAP formatting
  const getChartData = () => {
    if (!report || !report.attributions) return [];
    const data = [];
    
    report.attributions.top_risk_factors.forEach(f => {
      data.push({
        name: getReadableName(f.feature),
        val: f.value,
        type: 'risk'
      });
    });

    report.attributions.top_saving_factors.forEach(f => {
      data.push({
        name: getReadableName(f.feature),
        val: f.value,
        type: 'saving'
      });
    });

    return data.sort((a, b) => Math.abs(b.val) - Math.abs(a.val));
  };

  const chartData = getChartData();

  const suggestions = [
    `Assess credit risk for client ${selectedId || '100002'}`,
    "What documents do I need to submit if I am self-employed?",
    "Calculate EMI of 15 lakhs at 8.5% for 6 years"
  ];

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header glass-panel">
        <NeuroFinanceLogo />
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <button 
            className="suggestion-btn" 
            onClick={() => setTheme(prev => prev === 'dark' ? 'black' : prev === 'black' ? 'light' : 'dark')}
            style={{ 
              fontSize: '12px', 
              padding: '8px 16px', 
              borderRadius: '20px', 
              border: '1px solid var(--border-glass)',
              background: 'rgba(255,255,255,0.02)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              color: 'var(--text-primary)',
              transition: 'all 0.2s'
            }}
          >
            🎨 Theme: {theme === 'dark' ? 'Dark' : theme === 'black' ? 'OLED Black' : 'Light'}
          </button>
          <div className="status-badge">
            <span className={`status-dot ${backendStatus === 'online' ? 'active' : ''}`}></span>
            <span>Core System: {backendStatus === 'online' ? 'ONLINE' : 'OFFLINE'}</span>
          </div>
        </div>
      </header>

      {/* Selector Panel */}
      <section className="controls-panel glass-panel">
        <div className="select-container">
          <UserCheck size={18} style={{ color: 'var(--accent-purple)' }} />
          <span className="select-label">Select Borrower Profile:</span>
          <select 
            className="customer-select" 
            value={selectedId} 
            onChange={(e) => setSelectedId(e.target.value)}
            disabled={customers.length === 0}
          >
            {customers.length === 0 ? (
              <option>Connecting to Neural Engine backend...</option>
            ) : (
              customers.map(c => (
                <option key={c.sk_id_curr} value={c.sk_id_curr}>
                  Client ID: {c.sk_id_curr} | {c.gender === 'F' ? 'Female' : 'Male'} | Monthly Income: Rs. {Math.round(c.income/12).toLocaleString()} | Default Target: {c.target}
                </option>
              ))
            )}
          </select>
        </div>
      </section>

      {/* Main Grid */}
      <main className="main-layout">
        
        {/* Left Column */}
        <div className="dashboard-content">
          
          {/* Top cards */}
          <div className="metrics-grid">
            
            {/* Probability Card */}
            <div className="metric-card glass-panel">
              <div className="metric-icon-container" style={{ 
                color: (simulatedResult || report)?.risk_category === 'High' ? 'var(--color-danger)' : 
                       (simulatedResult || report)?.risk_category === 'Medium' ? 'var(--color-warning)' : 
                       'var(--color-success)' 
              }}>
                <ShieldAlert size={26} />
              </div>
              <div className="metric-info">
                <h3>Default Probability</h3>
                <p className="metric-value">
                  {report ? `${((simulatedResult?.default_probability ?? report.default_probability) * 100).toFixed(2)}%` : '0.00%'}
                </p>
                <p className="metric-label">
                  {simulatedResult ? '⚡ Modified Scenario Prediction' : 'Baseline Credit Score'}
                </p>
              </div>
            </div>

            {/* Financial Health Score Card */}
            <div className="metric-card glass-panel">
              <div className="metric-icon-container" style={{ color: 'var(--accent-purple)' }}>
                <Heart size={26} />
              </div>
              <div className="metric-info">
                <h3>Financial Health</h3>
                <p className="metric-value">
                  {report ? `${(simulatedResult?.financial_health_score ?? report.financial_health_score)}/100` : '0/100'}
                </p>
                <p className="metric-label">
                  {simulatedResult ? '⚡ Simulated health index' : 'Verified affordability index'}
                </p>
              </div>
            </div>

            {/* Risk Category Card */}
            <div className="metric-card glass-panel">
              <div className="metric-icon-container" style={{ 
                color: (simulatedResult || report)?.risk_category === 'High' ? 'var(--color-danger)' : 
                       (simulatedResult || report)?.risk_category === 'Medium' ? 'var(--color-warning)' : 
                       'var(--color-success)' 
              }}>
                <Activity size={26} />
              </div>
              <div className="metric-info">
                <h3>Approval Category</h3>
                <p className="metric-value" style={{ 
                  color: (simulatedResult || report)?.risk_category === 'High' ? 'var(--color-danger)' : 
                         (simulatedResult || report)?.risk_category === 'Medium' ? 'var(--color-warning)' : 
                         'var(--color-success)' 
                }}>
                  {report ? (simulatedResult?.risk_category ?? report.risk_category).toUpperCase() : 'N/A'}
                </p>
                <p className="metric-label">Lending recommendation</p>
              </div>
            </div>

          </div>

          {/* Interactive tabs */}
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column' }}>
            <div className="dashboard-tabs">
              <button 
                className={`tab-btn ${activeTab === 'risk' ? 'active' : ''}`}
                onClick={() => setActiveTab('risk')}
              >
                Risk Explanation (SHAP Matrix)
              </button>
              <button 
                className={`tab-btn ${activeTab === 'whatif' ? 'active' : ''}`}
                onClick={() => setActiveTab('whatif')}
              >
                What-If Risk Simulator
              </button>
              <button 
                className={`tab-btn ${activeTab === 'documents' ? 'active' : ''}`}
                onClick={() => setActiveTab('documents')}
              >
                Document Intelligence
              </button>
              <button 
                className={`tab-btn ${activeTab === 'profile' ? 'active' : ''}`}
                onClick={() => setActiveTab('profile')}
              >
                Applicant Profile Details
              </button>
            </div>

            <div className="tab-content">
              {loading ? (
                <div style={{ textAlign: 'center', padding: '50px 0', color: 'var(--text-secondary)' }}>
                  <Activity size={36} className="pulse" style={{ margin: '0 auto 16px', color: 'var(--accent-purple)' }} />
                  <p>Processing neural network embeddings and attributions...</p>
                </div>
              ) : error ? (
                <div style={{ textAlign: 'center', padding: '30px 0', color: 'var(--color-danger)' }}>
                  <AlertCircle size={36} style={{ margin: '0 auto 12px' }} />
                  <p>{error}</p>
                </div>
              ) : report ? (
                <>
                  {/* SHAP Attributions Tab */}
                  {activeTab === 'risk' && (
                    <div>
                      <h3 style={{ margin: '0 0 20px', fontSize: '16px', fontWeight: 600 }}>SHAP Credit Attribution Matrix</h3>
                      
                      <div style={{ width: '100%', height: 420 }}>
                        <ResponsiveContainer>
                          <BarChart
                            data={chartData}
                            layout="vertical"
                            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                          >
                            <defs>
                              <linearGradient id="riskGrad" x1="0" y1="0" x2="1" y2="0">
                                <stop offset="0%" stopColor="#ef4444" stopOpacity={0.65}/>
                                <stop offset="100%" stopColor="#f87171" stopOpacity={1}/>
                              </linearGradient>
                              <linearGradient id="savingGrad" x1="1" y1="0" x2="0" y2="0">
                                <stop offset="0%" stopColor="#10b981" stopOpacity={0.65}/>
                                <stop offset="100%" stopColor="#34d399" stopOpacity={1}/>
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(255,255,255,0.03)" />
                            <XAxis type="number" stroke="var(--text-secondary)" fontSize={12} />
                            <YAxis dataKey="name" type="category" stroke="var(--text-secondary)" fontSize={10} width={220} />
                            <Tooltip 
                              contentStyle={{ 
                                background: 'rgba(11, 11, 18, 0.95)', 
                                border: '1px solid rgba(192, 132, 252, 0.25)', 
                                borderRadius: '12px',
                                boxShadow: '0 10px 30px rgba(0,0,0,0.6)'
                              }}
                              itemStyle={{ color: '#fff', fontSize: '13px', fontFamily: 'var(--font-mono)' }}
                              labelStyle={{ color: 'var(--accent-purple)', fontWeight: 600, fontSize: '12px', marginBottom: '4px' }}
                            />
                            <ReferenceLine x={0} stroke="rgba(255,255,255,0.2)" strokeWidth={1.5} />
                            <Bar dataKey="val" radius={5}>
                              {chartData.map((entry, idx) => (
                                <Cell 
                                  key={`cell-${idx}`} 
                                  fill={entry.val > 0 ? 'url(#riskGrad)' : 'url(#savingGrad)'} 
                                />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>

                      <div className="factors-container">
                        <div className="factors-column">
                          <h4 style={{ color: 'var(--color-danger)' }}>
                            <AlertCircle size={16} /> Features Increasing Risk
                          </h4>
                          {report.attributions.top_risk_factors.map((f, i) => (
                            <div className="factor-item" key={i}>
                              <div className="factor-header">
                                <span className="factor-name">{getReadableName(f.feature)}</span>
                                <span>+{f.value.toFixed(4)}</span>
                              </div>
                              <div className="factor-bar-bg">
                                <div className="factor-bar-fill" style={{ width: `${Math.min(f.value * 200, 100)}%`, backgroundColor: 'var(--color-danger)' }}></div>
                              </div>
                            </div>
                          ))}
                        </div>

                        <div className="factors-column">
                          <h4 style={{ color: 'var(--color-success)' }}>
                            <CheckCircle2 size={16} /> Features Reducing Risk
                          </h4>
                          {report.attributions.top_saving_factors.map((f, i) => (
                            <div className="factor-item" key={i}>
                              <div className="factor-header">
                                <span className="factor-name">{getReadableName(f.feature)}</span>
                                <span>{f.value.toFixed(4)}</span>
                              </div>
                              <div className="factor-bar-bg">
                                <div className="factor-bar-fill" style={{ width: `${Math.min(Math.abs(f.value) * 200, 100)}%`, backgroundColor: 'var(--color-success)' }}></div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* What-If Simulator Tab */}
                  {activeTab === 'whatif' && (
                    <div className="simulation-panel">
                      <h3 style={{ margin: '0 0 10px', fontSize: '16px', fontWeight: 600 }}>What-If Borrower Scenario Simulator</h3>
                      <p style={{ fontSize: '13px', color: 'var(--text-secondary)', margin: '0 0 20px 0' }}>
                        Modify key financial parameters of the borrower profile in real-time. The system will process the modified attributes through our neural network to compute simulated risk scores.
                      </p>

                      <div className="sim-grid">
                        <div className="slider-group">
                          <div className="slider-header">
                            <span className="slider-name">Monthly Income (Rs.)</span>
                            <span className="slider-val">Rs. {Math.round(simInputs.income).toLocaleString()}</span>
                          </div>
                          <input 
                            type="range" 
                            min="20000" 
                            max="500000" 
                            step="5000"
                            className="slider-input"
                            value={simInputs.income} 
                            onChange={(e) => setSimInputs({ ...simInputs, income: parseFloat(e.target.value) })}
                          />
                        </div>

                        <div className="slider-group">
                          <div className="slider-header">
                            <span className="slider-name">Requested Credit Loan Amount (Rs.)</span>
                            <span className="slider-val">Rs. {Math.round(simInputs.credit).toLocaleString()}</span>
                          </div>
                          <input 
                            type="range" 
                            min="50000" 
                            max="2000000" 
                            step="10000"
                            className="slider-input"
                            value={simInputs.credit} 
                            onChange={(e) => setSimInputs({ ...simInputs, credit: parseFloat(e.target.value) })}
                          />
                        </div>

                        <div className="slider-group">
                          <div className="slider-header">
                            <span className="slider-name">Purchased Goods Price (Rs.)</span>
                            <span className="slider-val">Rs. {Math.round(simInputs.goods).toLocaleString()}</span>
                          </div>
                          <input 
                            type="range" 
                            min="50000" 
                            max="2000000" 
                            step="10000"
                            className="slider-input"
                            value={simInputs.goods} 
                            onChange={(e) => setSimInputs({ ...simInputs, goods: parseFloat(e.target.value) })}
                          />
                        </div>

                        <div className="slider-group">
                          <div className="slider-header">
                            <span className="slider-name">Bureau External Rating Score</span>
                            <span className="slider-val">{(simInputs.extSource).toFixed(3)} / 1.00</span>
                          </div>
                          <input 
                            type="range" 
                            min="0.01" 
                            max="0.99" 
                            step="0.01"
                            className="slider-input"
                            value={simInputs.extSource} 
                            onChange={(e) => setSimInputs({ ...simInputs, extSource: parseFloat(e.target.value) })}
                          />
                        </div>
                      </div>

                      <div className="sim-btn-wrapper">
                        <button 
                          className="calc-btn" 
                          onClick={handleRunSimulation}
                          disabled={simulating}
                          style={{ display: 'flex', alignItems: 'center', gap: 10 }}
                        >
                          <RefreshCw size={16} className={simulating ? 'pulse' : ''} />
                          {simulating ? 'Evaluating Profile...' : 'Simulate Loan Application'}
                        </button>
                      </div>

                      {simulatedResult && (
                        <div className="sim-comparison-widget animate-fade-in">
                          <div className="comparison-box">
                            <span className="comparison-label">Baseline Probability</span>
                            <span className="comparison-val" style={{ color: 'var(--text-secondary)' }}>
                              {(report.default_probability * 100).toFixed(2)}%
                            </span>
                          </div>
                          
                          <div className="comparison-box" style={{ borderLeft: '1px solid var(--border-glass)', borderRight: '1px solid var(--border-glass)' }}>
                            <span className="comparison-label">Simulated Probability</span>
                            <span className="comparison-val" style={{ 
                              color: simulatedResult.risk_category === 'High' ? 'var(--color-danger)' : 
                                     simulatedResult.risk_category === 'Medium' ? 'var(--color-warning)' : 
                                     'var(--color-success)' 
                            }}>
                              {(simulatedResult.default_probability * 100).toFixed(2)}%
                            </span>
                          </div>

                          <div className="comparison-box">
                            <span className="comparison-label">Net Deviation</span>
                            {(() => {
                              const delta = simulatedResult.default_probability - report.default_probability;
                              const isReduced = delta < 0;
                              return (
                                <>
                                  <span className="comparison-val" style={{ color: isReduced ? 'var(--color-success)' : 'var(--color-danger)' }}>
                                    {isReduced ? '' : '+'}{(delta * 100).toFixed(2)}%
                                  </span>
                                  <span className="comparison-delta" style={{ color: isReduced ? 'var(--color-success)' : 'var(--color-danger)' }}>
                                    {isReduced ? <TrendingDown size={14} /> : <TrendingUp size={14} />}
                                    {isReduced ? 'Risk Reduced' : 'Risk Increased'}
                                  </span>
                                </>
                              );
                            })()}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Document Intelligence Tab */}
                  {activeTab === 'documents' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                      <h3 style={{ margin: '0 0 10px', fontSize: '16px', fontWeight: 600 }}>CNN Document Intelligence Classifier & Verification</h3>
                      
                      {uploading ? (
                        <div style={{ textAlign: 'center', padding: '40px 0', color: 'var(--text-secondary)' }}>
                          <Activity size={32} className="pulse" style={{ margin: '0 auto 12px', color: 'var(--accent-purple)' }} />
                          <p>Running CNN Document Classifier and OCR text parsers...</p>
                        </div>
                      ) : !uploadedFile ? (
                        <div className="upload-box">
                          <Upload size={36} style={{ color: 'var(--accent-purple)' }} />
                          <div style={{ fontSize: '14px', fontWeight: 500 }}>Upload Borrower Verification Document</div>
                          <p style={{ fontSize: '12px', color: 'var(--text-secondary)', maxWidth: '400px', margin: '0' }}>
                            Upload an image format of a payslip, bank statement, or loan circular (JPG/PNG). The CNN model will classify the document type and run OCR extraction.
                          </p>
                          <input 
                            type="file" 
                            id="doc-upload-input" 
                            accept=".jpg,.jpeg,.png"
                            onChange={handleFileUpload} 
                            style={{ display: 'none' }} 
                          />
                          <label htmlFor="doc-upload-input" className="calc-btn" style={{ cursor: 'pointer', padding: '10px 20px', fontSize: '13px' }}>
                            Choose Document File
                          </label>
                        </div>
                      ) : (
                        <div className="upload-container animate-fade-in">
                          {/* Left Column - Image */}
                          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                            <img 
                              src={`${API_URL}/static/${uploadedFile.filename}`} 
                              alt="Document Preview" 
                              className="document-preview-img"
                            />
                            <button 
                              className="suggestion-btn" 
                              onClick={() => setUploadedFile(null)}
                              style={{ alignSelf: 'flex-start', fontSize: '11px', color: 'var(--text-primary)' }}
                            >
                              Upload a Different Document
                            </button>
                          </div>
                          
                          {/* Right Column - Results */}
                          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-glass)', paddingBottom: '12px' }}>
                              <div>
                                <span style={{ fontSize: '11px', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Classification</span>
                                <h4 style={{ margin: '2px 0 0', fontSize: '18px', fontWeight: 700, color: 'var(--accent-purple)' }}>
                                  {uploadedFile.document_type}
                                </h4>
                              </div>
                              <span className="logo-badge" style={{ backgroundColor: 'rgba(52,211,153,0.1)', color: 'var(--color-success)', borderColor: 'rgba(52,211,153,0.2)' }}>
                                Confidence: {(uploadedFile.confidence * 100).toFixed(2)}%
                              </span>
                            </div>

                            {uploadedFile.validation_warnings.length > 0 && (
                              <div className="warning-callout">
                                <AlertCircle size={16} />
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                                  {uploadedFile.validation_warnings.map((w, idx) => (
                                    <span key={idx}>{w}</span>
                                  ))}
                                </div>
                              </div>
                            )}

                            <div>
                              <span style={{ fontSize: '11px', color: 'var(--text-secondary)', textTransform: 'uppercase', display: 'block', marginBottom: '8px' }}>Extracted Metadata</span>
                              <div className="profile-grid" style={{ gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                                {Object.entries(uploadedFile.extracted_fields).map(([k, v]) => (
                                  <div className="profile-field" style={{ padding: '10px 14px' }} key={k}>
                                    <span className="field-label" style={{ fontSize: '9px', marginBottom: '4px' }}>{getReadableName(k)}</span>
                                    <span className="field-value" style={{ fontSize: '13px' }}>{formatValue(k, v)}</span>
                                  </div>
                                ))}
                              </div>
                            </div>

                            <button 
                              className="calc-btn" 
                              onClick={handleAnalyzeDocRisk}
                              disabled={analyzingDoc}
                              style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10, marginTop: '10px' }}
                            >
                              <FileCheck size={16} />
                              {analyzingDoc ? 'Simulating Credit Scores...' : 'Analyze Financial Risk'}
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Customer Profile Tab */}
                  {activeTab === 'profile' && (
                    <div className="profile-grid">
                      {Object.entries(report.profile)
                        .filter(([k]) => !['SK_ID_CURR', 'TARGET'].includes(k))
                        .slice(0, 24)
                        .map(([key, val]) => (
                          <div className="profile-field" key={key}>
                            <span className="field-label">{getReadableName(key)}</span>
                            <span className="field-value">{formatValue(key, val)}</span>
                          </div>
                        ))}
                    </div>
                  )}
                </>
              ) : (
                <p style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>Select a borrower profile to begin analysis.</p>
              )}
            </div>
          </div>

          {/* EMI Loan Calculator & Advisor */}
          <div className="glass-panel tab-content affordability-panel">
            <h3 style={{ margin: '0', fontSize: '16px', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 10 }}>
              <Calculator size={18} style={{ color: 'var(--accent-purple)' }} />
              EMI Loan Calculator & Affordability Advisory Planner
            </h3>
            
            <div className="calculator-form">
              <div className="form-group">
                <label>Principal Amount (Rs.)</label>
                <input 
                  type="number" 
                  className="form-input" 
                  value={emiPrincipal} 
                  onChange={(e) => setEmiPrincipal(e.target.value)} 
                />
              </div>
              <div className="form-group">
                <label>Annual Interest Rate (%)</label>
                <input 
                  type="number" 
                  step="0.1" 
                  className="form-input" 
                  value={emiRate} 
                  onChange={(e) => setEmiRate(e.target.value)} 
                />
              </div>
              <div className="form-group">
                <label>Tenure (Months)</label>
                <input 
                  type="number" 
                  className="form-input" 
                  value={emiMonths} 
                  onChange={(e) => setEmiMonths(e.target.value)} 
                />
              </div>
              <div className="form-group" style={{ justifyContent: 'flex-end', flexDirection: 'row', gap: '10px' }}>
                <button 
                  className="suggestion-btn" 
                  onClick={prefillAffordableEMI}
                  style={{ alignSelf: 'flex-end', padding: '12px 18px', fontSize: '13px', borderRadius: 'var(--radius-md)', whiteSpace: 'nowrap' }}
                >
                  Target Affordable (30% DTI)
                </button>
                <button className="calc-btn" onClick={handleCalculateEMI}>Compute Payments</button>
              </div>
            </div>

            {emiResult && (
              <div className="calc-results">
                <div className="result-box">
                  <div className="result-title">Monthly EMI</div>
                  <div className="result-value" style={{ color: 'var(--accent-purple)' }}>Rs. {emiResult.emi.toLocaleString()}</div>
                </div>
                <div className="result-box">
                  <div className="result-title">Total Interest</div>
                  <div className="result-value" style={{ color: 'var(--text-primary)' }}>Rs. {emiResult.total_interest.toLocaleString()}</div>
                </div>
                <div className="result-box">
                  <div className="result-title">Total Repayment</div>
                  <div className="result-value" style={{ color: 'var(--accent-blue)' }}>Rs. {emiResult.total_payment.toLocaleString()}</div>
                </div>
              </div>
            )}

            {/* Affordability Advisory Card */}
            {dtiDetails && (
              <div className="dti-card animate-fade-in">
                <div className="dti-header">
                  <span className="dti-title">Debt-to-Income (DTI) Annuity Ratio</span>
                  <span className={`dti-badge ${dtiDetails.status}`}>
                    {dtiDetails.badgeText} ({dtiDetails.dti}%)
                  </span>
                </div>
                
                <div className="dti-bar-container">
                  <div 
                    className="dti-bar-fill" 
                    style={{ 
                      width: `${Math.min(dtiDetails.dti, 100)}%`,
                      backgroundColor: dtiDetails.status === 'critical' ? 'var(--color-danger)' : 
                                      dtiDetails.status === 'stretched' ? 'var(--color-warning)' : 
                                      'var(--color-success)' 
                    }}
                  />
                  <div className="dti-marker" style={{ left: '50%' }} />
                </div>
                
                <p className="dti-description">
                  {dtiDetails.advisory}
                </p>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between', borderTop: '1px solid rgba(255,255,255,0.03)', paddingTop: '10px', marginTop: '6px' }}>
                  <span>Verified Borrower Monthly Net Income: <strong>Rs. {dtiDetails.monthlyIncome.toLocaleString()}</strong></span>
                  <span>Calculated Installment: <strong>Rs. {emiResult?.emi.toLocaleString()} / mo</strong></span>
                </div>
              </div>
            )}
          </div>

        </div>

        {/* Right Column: Conversational Chatbot */}
        <aside className="chat-sidebar glass-panel chat-container">
          <div className="chat-header">
            <div className="chat-header-avatar">
              <Sparkles size={20} />
            </div>
            <div className="chat-header-text">
              <h3>NeuroBot Decision Assistant</h3>
              <p>Risk Router • Policy RAG • Loan Math</p>
            </div>
          </div>

          {/* Glowing Float Robot Mascot Widget */}
          <div className="robot-mascot-wrapper">
            <RobotMascot />
            <div className="robot-speech-bubble">
              {chatLoading ? "Formulating policy guidance..." :
               loading ? "Accessing neural net layers..." :
               (simulatedResult?.default_probability ?? report?.prediction?.default_probability) > 0.15 ? "Caution: Default risk is elevated! ⚠️" :
               (simulatedResult?.default_probability ?? report?.prediction?.default_probability) > 0 ? "Underwriting profile evaluated! 📊" :
               "Welcome! Ready for credit advisory. 🚀"}
            </div>
          </div>

          <div className="chat-messages">
            {messages.map((m, i) => (
              <div className={`chat-bubble-wrapper ${m.sender}`} key={i}>
                <div className={`chat-avatar ${m.sender}`}>
                  {m.sender === 'user' ? 'U' : '🤖'}
                </div>
                <div className={`chat-bubble ${m.sender}`}>
                  {renderFormattedText(m.text)}
                </div>
              </div>
            ))}
            {chatLoading && (
              <div className="chat-bubble-wrapper bot pulse">
                <div className="chat-avatar bot">
                  <Activity size={14} className="pulse" />
                </div>
                <div className="chat-bubble bot" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span>NeuroBot is formulating response...</span>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Suggestions */}
          <div className="chat-suggestions">
            <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: '2px', letterSpacing: '0.5px' }}>Helpful Prompts:</span>
            {suggestions.map((s, idx) => (
              <button 
                key={idx} 
                className="suggestion-btn"
                onClick={() => handleSendMessage(s)}
                disabled={chatLoading}
              >
                {s}
              </button>
            ))}
          </div>

          <div className="chat-input-area">
            <input 
              type="text" 
              className="chat-input"
              placeholder="Ask a loan policy or check risk score..."
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
              disabled={chatLoading}
            />
            <button 
              className="send-btn" 
              onClick={() => handleSendMessage()}
              disabled={chatLoading}
            >
              <Send size={16} />
            </button>
          </div>
        </aside>

      </main>
    </div>
  );
}

export default App;
