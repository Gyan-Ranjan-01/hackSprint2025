// ==================== DEPENDENCIES ====================
require('dotenv').config();
const express = require("express");
const app = express();
const path = require("path");
const ejsMate = require("ejs-mate");
const methodOverride = require("method-override");
const mongoose = require("mongoose");
const User = require("./models/user.js");
const session = require("express-session");
const cors = require('cors');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const Groq = require('groq-sdk');

const dbUrl = process.env.MONGO_URL || "mongodb://127.0.0.1:27017/hacksprint";

// ==================== MODEL CONFIGURATION ====================
// Define model fallback order with Groq integration
const MODEL_PRIORITY = [
    {
        name: "gemini-2.5-flash",
        provider: "gemini",
        config: { temperature: 0.7, topK: 40, topP: 0.95, maxOutputTokens: 8000 }
    },
    {
        name: "gemini-2.5-flash-lite", 
        provider: "gemini",
        config: { temperature: 0.7, topK: 40, topP: 0.95, maxOutputTokens: 8000 }
    },
    {
        name: "llama-3.3-70b-versatile",
        provider: "groq",
        config: { temperature: 0.7, max_tokens: 8000 }
    },
    {
        name: "gemini-2.5-flash-tts",
        provider: "gemini",
        config: { temperature: 0.7, topK: 40, topP: 0.95, maxOutputTokens: 800 }
    },
    {
        name: "gemini-3-flash",
        provider: "gemini",
        config: { temperature: 0.7, topK: 40, topP: 0.95, maxOutputTokens: 8000 }
    },
    {
        name: "gemma-3-12b",
        provider: "gemini",
        config: { temperature: 0.7, topK: 40, topP: 0.95, maxOutputTokens: 8000 }
    },
    {
        name: "gemma-3-1b",
        provider: "gemini",
        config: { temperature: 0.7, topK: 40, topP: 0.95, maxOutputTokens: 8000 }
    },
    {
        name: "gemma-3-27b",
        provider: "gemini",
        config: { temperature: 0.7, topK: 40, topP: 0.95, maxOutputTokens: 8000 }
    },
    {
        name: "gemma-3-2b",
        provider: "gemini",
        config: { temperature: 0.7, topK: 40, topP: 0.95, maxOutputTokens: 8000 }
    },
    {
        name: "gemma-3-4b",
        provider: "gemini",
        config: { temperature: 0.7, topK: 40, topP: 0.95, maxOutputTokens: 8000 }
    },
    {
        name: "gemini-robotics-er-1.5-preview",
        provider: "gemini",
        config: { temperature: 0.7, topK: 40, topP: 0.95, maxOutputTokens: 8000 }
    }
];

// Track model usage and failures
const modelStats = new Map();
MODEL_PRIORITY.forEach(model => {
    modelStats.set(model.name, { 
        attempts: 0, 
        failures: 0, 
        lastFailure: null,
        rateLimitHits: 0 
    });
});

// ==================== UTILITY FUNCTIONS ====================
// Wrap async functions to catch errors
const wrapAsync = (fn) => {
    return (req, res, next) => {
        Promise.resolve(fn(req, res, next)).catch(next);
    };
};

// Custom error class
class AppError extends Error {
    constructor(message, statusCode) {
        super(message);
        this.statusCode = statusCode;
        this.isOperational = true;
    }
}

// AI generation with automatic fallback (supports both Gemini and Groq)
async function generateWithFallback(genAI, groqClient, prompt, customConfig = {}, imageParts = null) {
    let lastError = null;
    
    for (const modelConfig of MODEL_PRIORITY) {
        const stats = modelStats.get(modelConfig.name);
        stats.attempts++;
        
        // Skip if recently rate limited (within last 60 seconds)
        if (stats.lastFailure && (Date.now() - stats.lastFailure) < 60000) {
            console.log(`⏭️ Skipping ${modelConfig.name} (recently rate limited)`);
            continue;
        }
        
        try {
            console.log(`🤖 Trying model: ${modelConfig.name} (${modelConfig.provider}) - Attempt ${stats.attempts}`);
            
            if (modelConfig.provider === "groq") {
                // Use Groq API
                if (imageParts) {
                    console.log(`⚠️ Groq doesn't support image input, skipping to next model...`);
                    continue;
                }
                
                const completion = await groqClient.chat.completions.create({
                    model: modelConfig.name,
                    messages: [
                        {
                            role: "user",
                            content: prompt
                        }
                    ],
                    temperature: customConfig.temperature || modelConfig.config.temperature,
                    max_tokens: customConfig.maxOutputTokens || modelConfig.config.max_tokens,
                });
                
                const text = completion.choices[0]?.message?.content || '';
                console.log(`✅ Success with ${modelConfig.name} (Groq)`);
                
                return { 
                    text, 
                    modelUsed: modelConfig.name,
                    provider: 'groq',
                    success: true 
                };
                
            } else {
                // Use Gemini API
                const model = genAI.getGenerativeModel({
                    model: modelConfig.name,
                    generationConfig: { ...modelConfig.config, ...customConfig }
                });
                
                let result;
                if (imageParts) {
                    result = await model.generateContent([prompt, ...imageParts]);
                } else {
                    result = await model.generateContent(prompt);
                }
                
                const text = result.response.text();
                console.log(`✅ Success with ${modelConfig.name} (Gemini)`);
                
                return { 
                    text, 
                    modelUsed: modelConfig.name,
                    provider: 'gemini',
                    success: true 
                };
            }
            
        } catch (error) {
            stats.failures++;
            lastError = error;
            
            // Check if it's a rate limit error
            const isRateLimit = error.message?.includes('RESOURCE_EXHAUSTED') || 
                               error.message?.includes('429') ||
                               error.message?.includes('quota') ||
                               error.message?.includes('rate limit') ||
                               error.message?.includes('Rate limit');
            
            if (isRateLimit) {
                stats.rateLimitHits++;
                stats.lastFailure = Date.now();
                console.log(`⚠️ Rate limit hit on ${modelConfig.name}. Trying next model...`);
            } else {
                console.log(`❌ Error with ${modelConfig.name}: ${error.message}`);
            }
            
            // Continue to next model
            continue;
        }
    }
    
    // All models failed
    console.error('❌ All models failed');
    throw new AppError(
        `AI service temporarily unavailable. Please try again in a moment. Last error: ${lastError?.message || 'Unknown'}`,
        503
    );
}

// Chat generation with fallback (supports both Gemini and Groq)
async function chatWithFallback(genAI, groqClient, chat, message, sessionId) {
    let lastError = null;
    
    for (const modelConfig of MODEL_PRIORITY) {
        const stats = modelStats.get(modelConfig.name);
        
        if (stats.lastFailure && (Date.now() - stats.lastFailure) < 60000) {
            continue;
        }
        
        try {
            console.log(`🤖 Chat trying: ${modelConfig.name} (${modelConfig.provider})`);
            stats.attempts++;
            
            if (modelConfig.provider === "groq") {
                // Use Groq for chat
                const systemMessage = "You are Dr. AI, the official medical assistant for the DocOnCall platform. Your role is to be a friendly and empathetic medical assistant chatbot for a virtual healthcare platform. Your role is to: 1) Ask relevant questions about symptoms, 2) Provide general health guidance, 3) Show empathy and be reassuring, 4) Keep responses concise (2-4 sentences), 5) ALWAYS remind users this is not a replacement for professional medical advice. Be warm, professional, and helpful.";
                
                const completion = await groqClient.chat.completions.create({
                    model: modelConfig.name,
                    messages: [
                        {
                            role: "system",
                            content: systemMessage
                        },
                        {
                            role: "user",
                            content: message
                        }
                    ],
                    temperature: modelConfig.config.temperature,
                    max_tokens: modelConfig.config.max_tokens,
                });
                
                const reply = completion.choices[0]?.message?.content || '';
                console.log(`✅ Chat success with ${modelConfig.name} (Groq)`);
                
                return { 
                    reply, 
                    chat: null, // Groq doesn't maintain chat state in the same way
                    modelUsed: modelConfig.name,
                    provider: 'groq',
                    success: true 
                };
                
            } else {
                // Use Gemini for chat
                // If we need to recreate chat with different model
                if (!chat || chat.modelName !== modelConfig.name) {
                    const model = genAI.getGenerativeModel({
                        model: modelConfig.name,
                        generationConfig: modelConfig.config
                    });
                    
                    chat = model.startChat({
                        history: [
                            {
                                role: "user",
                                parts: [{ text: "You are Dr. AI, the official medical assistant for the DocOnCall platform. Your role is to be a friendly and empathetic medical assistant chatbot for a virtual healthcare platform. Your role is to: 1) Ask relevant questions about symptoms, 2) Provide general health guidance, 3) Show empathy and be reassuring, 4) Keep responses concise (2-4 sentences), 5) ALWAYS remind users this is not a replacement for professional medical advice. Be warm, professional, and helpful." }]
                            },
                            {
                                role: "model",
                                parts: [{ text: "Hello! I'm Dr. AI, your virtual health assistant. I'm here to help answer your health questions and provide general guidance. Please remember that I'm not a replacement for professional medical advice. How can I assist you today?" }]
                            }
                        ]
                    });
                    chat.modelName = modelConfig.name;
                }
                
                const result = await chat.sendMessage(message);
                const reply = result.response.text();
                
                console.log(`✅ Chat success with ${modelConfig.name} (Gemini)`);
                
                return { 
                    reply, 
                    chat,
                    modelUsed: modelConfig.name,
                    provider: 'gemini',
                    success: true 
                };
            }
            
        } catch (error) {
            stats.failures++;
            lastError = error;
            
            const isRateLimit = error.message?.includes('RESOURCE_EXHAUSTED') || 
                               error.message?.includes('429') ||
                               error.message?.includes('quota') ||
                               error.message?.includes('rate limit') ||
                               error.message?.includes('Rate limit');
            
            if (isRateLimit) {
                stats.rateLimitHits++;
                stats.lastFailure = Date.now();
                console.log(`⚠️ Chat rate limit on ${modelConfig.name}`);
            }
            
            continue;
        }
    }
    
    throw new AppError(
        `AI chat temporarily unavailable. Please try again in a moment.`,
        503
    );
}

// ==================== MIDDLEWARE SETUP ====================
app.engine("ejs", ejsMate);
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));
app.use(express.static(path.join(__dirname, "public")));
app.use(express.urlencoded({ extended: true }));
app.use(express.json({ limit: '50mb' }));
app.use(methodOverride("_method"));
app.use(cors());

// ==================== SESSION CONFIGURATION ====================
const sessionConfig = {
    secret: process.env.SESSION_SECRET || "fallbackSecret",
    resave: false,
    saveUninitialized: false,
    cookie: {
        httpOnly: true,
        expires: Date.now() + 1000 * 60 * 60 * 24 * 7,
        maxAge: 1000 * 60 * 60 * 24 * 7,
    },
};
app.use(session(sessionConfig));

// Make current user available to all templates
app.use((req, res, next) => {
    res.locals.currentUser = req.session.user;
    next();
});

// ==================== AI INITIALIZATION ====================
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const groqClient = new Groq({
    apiKey: process.env.GROQ_API_KEY
});
const chatSessions = new Map();

console.log('🤖 Initializing AI providers with fallback support...');
console.log(`🔑 Gemini API Key: ${process.env.GEMINI_API_KEY ? 'Configured ✓' : 'Missing ✗'}`);
console.log(`🔑 Groq API Key: ${process.env.GROQ_API_KEY ? 'Configured ✓' : 'Missing ✗'}`);
console.log('📋 Fallback order:', MODEL_PRIORITY.map(m => `${m.name} (${m.provider})`).join(' → '));

// ==================== MONGODB CONNECTION ====================
async function main() {
    console.log("Connecting to MongoDB...HackSprint");
    await mongoose.connect(dbUrl);
}

main()
    .then(() => {
        console.log("connected to mongoDB");
    })
    .catch((err) => {
        console.log(err);
    });

// ==================== AUTHENTICATION MIDDLEWARE ====================
const requireLogin = (req, res, next) => {
    if (!req.session.user_id) {
        return res.redirect('/login');
    }
    next();
};

// ==================== AUTHENTICATION ROUTES ====================

// Home route
app.get("/", (req, res) => {
    if (req.session.user_id) {
        res.render("main");
    } else {
        res.render("welcome");
    }
});

// Login routes
app.get("/login", (req, res) => {
    res.render("login");
});

app.post("/login", wrapAsync(async (req, res, next) => {
    const { email, password } = req.body;

    if (!email || !password) {
        throw new AppError("Email and password are required", 400);
    }

    const user = await User.findOne({ email });

    if (!user) {
        throw new AppError("Invalid email or password", 401);
    }

    const isValid = password === user.password;

    if (!isValid) {
        throw new AppError("Invalid email or password", 401);
    }

    req.session.user_id = user._id;
    req.session.user = {
        id: user._id,
        name: user.name,
        email: user.email
    };
    res.redirect("/");
}));

// Signup routes
app.get("/signup", (req, res) => {
    res.render("signup");
});

app.post("/signup", wrapAsync(async (req, res, next) => {
    const { name, email, password } = req.body;

    if (!name || !email || !password) {
        throw new AppError("All fields are required", 400);
    }

    const existingUser = await User.findOne({ email });
    if (existingUser) {
        throw new AppError("Email already registered", 409);
    }

    const user = new User({ name, email, password });
    await user.save();

    req.session.user_id = user._id;
    req.session.user = {
        id: user._id,
        name: user.name,
        email: user.email
    };

    res.redirect("/");
}));

// Logout route
app.get("/logout", (req, res) => {
    req.session.destroy((err) => {
        if (err) {
            return res.redirect("/");
        }
        res.redirect("/login");
    });
});

app.get("/help", (req, res) => {
    res.render("help");
});

// ==================== AI FEATURE PAGE ROUTES ====================

app.get("/index", requireLogin, (req, res) => {
    res.render("index");
});
app.get("/chatbot", requireLogin, (req, res) => {
    res.render("chatbot");
});

app.get("/symptom-analysis", requireLogin, (req, res) => {
    res.render("symptom-analysis");
});

app.get("/report-summary", requireLogin, (req, res) => {
    res.render("report-summary");
});

app.get("/medicine-info", requireLogin, (req, res) => {
    res.render("medical-info");
});

app.get("/health-tips", requireLogin, (req, res) => {
    res.render("health-tips");
});

app.get("/diet-plan", requireLogin, (req, res) => {
    res.render("diet-plan");
});

// ==================== AI HEALTHCARE ROUTES ====================

// 1. Health Chatbot
app.post('/api/chat', requireLogin, wrapAsync(async (req, res) => {
    const { message, sessionId = 'default' } = req.body;

    if (!message) {
        throw new AppError('Message is required', 400);
    }

    console.log(`💬 Chat [${sessionId}]: ${message}`);

    let chat = chatSessions.get(sessionId);
    const result = await chatWithFallback(genAI, groqClient, chat, message, sessionId);
    
    if (result.chat) {
        chatSessions.set(sessionId, result.chat);
    }

    res.json({
        reply: result.reply,
        success: true,
        sessionId,
        modelUsed: result.modelUsed,
        provider: result.provider
    });
}));

// 2. Symptom Analysis
app.post('/api/analyze-symptoms', requireLogin, wrapAsync(async (req, res) => {
    const { symptoms, age, gender, duration } = req.body;

    if (!symptoms) {
        throw new AppError('Symptoms are required', 400);
    }

    console.log('🔍 Analyzing symptoms:', symptoms);

    const prompt = `You are a medical AI assistant. Analyze the following patient symptoms and provide a structured medical assessment.

**Patient Information:**
- Age: ${age || 'Not specified'}
- Gender: ${gender || 'Not specified'}
- Symptoms: ${symptoms}
- Duration: ${duration || 'Not specified'}

**Provide analysis in this exact format:**

**POSSIBLE CONDITIONS:**
1. [Condition Name] - Probability: [High/Medium/Low]
   Brief explanation of why this is suspected.

2. [Condition Name] - Probability: [High/Medium/Low]
   Brief explanation of why this is suspected.

3. [Condition Name] - Probability: [High/Medium/Low]
   Brief explanation of why this is suspected.

**SEVERITY LEVEL:** [Low/Medium/High]
Brief explanation of severity assessment.

**URGENCY:** [Immediate/Soon/Routine]
When should the patient seek medical attention.

**RECOMMENDATIONS:**
1. [Specific recommendation]
2. [Specific recommendation]
3. [Specific recommendation]
4. [Specific recommendation]
5. [Specific recommendation]

**IMPORTANT DISCLAIMER:**
[Clear statement about seeking professional medical help]

Provide detailed, accurate, and helpful information while being clear this is preliminary guidance only.`;

    const result = await generateWithFallback(genAI, groqClient, prompt, {
        temperature: 0.3,
        maxOutputTokens: 600
    });

    console.log('✅ Analysis complete');

    res.json({
        analysis: result.text,
        success: true,
        modelUsed: result.modelUsed,
        provider: result.provider
    });
}));

// 3. Medical Report Summary
app.post('/api/summarize-report', requireLogin, wrapAsync(async (req, res) => {
    const { reportText, reportType } = req.body;

    if (!reportText) {
        throw new AppError('Report text is required', 400);
    }

    console.log('📄 Summarizing report...');

    const prompt = `You are a medical AI assistant specializing in explaining medical reports to patients.

**Report Type:** ${reportType || 'Medical Report'}

**Report Content:**
${reportText}

**Please provide a patient-friendly summary with:**

1. **KEY FINDINGS:** Main results from the report
2. **ABNORMAL VALUES:** Any values outside normal range (explain what they mean)
3. **WHAT IT MEANS:** Is not normal or what could be the implications and should the patient be concerned with doctor?
4. **NEXT STEPS:** Suggested actions or follow-up needed
5. **QUESTIONS TO ASK YOUR DOCTOR:** Important questions patient should ask

Use simple, non-technical language. Avoid medical jargon. Be clear and reassuring while being honest about findings.`;

    const result = await generateWithFallback(genAI, groqClient, prompt, {
        temperature: 0.4,
        maxOutputTokens: 500
    });

    console.log('✅ Summary generated');

    res.json({
        summary: result.text,
        success: true,
        modelUsed: result.modelUsed,
        provider: result.provider
    });
}));

// 4. Medicine Information
app.post('/api/medicine-info', requireLogin, wrapAsync(async (req, res) => {
    const { medicineName } = req.body;

    if (!medicineName) {
        throw new AppError('Medicine name is required', 400);
    }

    console.log('💊 Getting info for:', medicineName);

    const prompt = `Provide accurate, patient-friendly information about the medicine: **${medicineName}**

Include the following sections:

**WHAT IT'S USED FOR:**
Primary uses and conditions it treats.

**HOW IT WORKS:**
Simple explanation of mechanism.

**COMMON SIDE EFFECTS:**
Most frequently reported side effects.

**IMPORTANT PRECAUTIONS:**
- Who should not take it
- Drug interactions to be aware of
- Special warnings

**WHEN TO CONSULT A DOCTOR:**
Signs that require immediate medical attention.

**IMPORTANT NOTE:**
Always remind to consult healthcare provider or pharmacist for personalized advice.

Keep information accurate and helpful. Use simple language.`;

    const result = await generateWithFallback(genAI, groqClient, prompt, {
        temperature: 0.3,
        maxOutputTokens: 400
    });

    console.log('✅ Medicine info retrieved');

    res.json({
        info: result.text,
        success: true,
        modelUsed: result.modelUsed,
        provider: result.provider
    });
}));

// 5. Health Tips Generator
app.post('/api/health-tips', requireLogin, wrapAsync(async (req, res) => {
    const { category, userProfile } = req.body;

    console.log('💡 Generating tips for:', category);

    const profileStr = userProfile ? JSON.stringify(userProfile) : 'General audience';

    const prompt = `Generate 5 practical, evidence-based health tips for the category: **${category || 'General Health'}**

User Profile: ${profileStr}

**Format each tip as:**
**Tip #X: [Catchy Title]**
[Detailed explanation of the tip - 2-3 sentences]
Why it matters: [Brief benefit explanation]

Make tips:
- Actionable and specific
- Easy to implement in daily life
- Based on scientific evidence
- Motivating and positive
- Culturally sensitive

Provide exactly 5 tips, numbered 1-5.`;

    const result = await generateWithFallback(genAI, groqClient, prompt, {
        temperature: 0.7,
        maxOutputTokens: 500
    });

    console.log('✅ Tips generated');

    res.json({
        tips: result.text,
        success: true,
        modelUsed: result.modelUsed,
        provider: result.provider
    });
}));

// 6. Diet Plan Generator
app.post('/api/diet-plan', requireLogin, wrapAsync(async (req, res) => {
    const { goal, restrictions, preferences } = req.body;

    console.log('🥗 Generating diet plan for goal:', goal);

    const prompt = `Create a personalized one-day diet plan.

**Goal:** ${goal || 'General health'}
**Dietary Restrictions:** ${restrictions || 'None'}
**Preferences:** ${preferences || 'None'}

Provide:

**BREAKFAST (7-9 AM):**
- Meal description
- Approximate calories
- Key nutrients

**MID-MORNING SNACK (11 AM):**
- Snack suggestion
- Benefits

**LUNCH (1-2 PM):**
- Meal description
- Approximate calories
- Key nutrients

**EVENING SNACK (4-5 PM):**
- Snack suggestion
- Benefits

**DINNER (7-8 PM):**
- Meal description
- Approximate calories
- Key nutrients

**HYDRATION TIPS:**
Water intake recommendations

**NUTRITIONAL SUMMARY:**
Total approximate calories and macronutrient breakdown

Make it practical, affordable, and easy to prepare.`;

    const result = await generateWithFallback(genAI, groqClient, prompt, {
        temperature: 0.6,
        maxOutputTokens: 800
    });

    console.log('✅ Diet plan generated');

    res.json({
        plan: result.text,
        success: true,
        modelUsed: result.modelUsed,
        provider: result.provider
    });
}));

// 7. Prescription Image Analysis
app.post('/api/read-prescription', requireLogin, wrapAsync(async (req, res) => {
    const { imageBase64 } = req.body;

    if (!imageBase64) {
        throw new AppError('Image data is required', 400);
    }

    console.log('📸 Analyzing prescription image...');

    const prompt = `Analyze this prescription image and extract all readable information.

Provide:

**MEDICINES PRESCRIBED:**
List each medicine with:
- Medicine name
- Dosage (strength)
- Frequency (how often to take)
- Duration (how many days)

**SPECIAL INSTRUCTIONS:**
Any additional notes or warnings

**DOCTOR INFORMATION:**
Doctor's name and credentials if visible

**DATE:**
Prescription date if visible

**IMPORTANT NOTES:**
- Highlight any unclear or unreadable parts
- Note if prescription needs verification
- Remind to consult pharmacist if unsure

Be accurate and clear. If something is unclear, state it explicitly.`;

    const base64Data = imageBase64.includes(',')
        ? imageBase64.split(',')[1]
        : imageBase64;

    const imageParts = [
        {
            inlineData: {
                data: base64Data,
                mimeType: "image/jpeg"
            }
        }
    ];

    const result = await generateWithFallback(genAI, groqClient, prompt, {
        temperature: 0.2,
        maxOutputTokens: 500
    }, imageParts);

    console.log('✅ Prescription analyzed');

    res.json({
        analysis: result.text,
        success: true,
        modelUsed: result.modelUsed,
        provider: result.provider
    });
}));

// ==================== UTILITY ROUTES ====================

// Clear specific chat session
app.post('/api/clear-chat', requireLogin, (req, res) => {
    const { sessionId = 'default' } = req.body;
    chatSessions.delete(sessionId);
    console.log(`🗑️ Cleared chat session: ${sessionId}`);
    res.json({
        success: true,
        message: 'Chat history cleared',
        sessionId
    });
});

// Clear all chat sessions
app.post('/api/clear-all-chats', requireLogin, (req, res) => {
    const count = chatSessions.size;
    chatSessions.clear();
    console.log(`🗑️ Cleared all ${count} chat sessions`);
    res.json({
        success: true,
        message: `Cleared ${count} chat sessions`
    });
});

// Model stats endpoint
app.get('/api/model-stats', requireLogin, (req, res) => {
    const stats = {};
    modelStats.forEach((value, key) => {
        stats[key] = {
            ...value,
            successRate: value.attempts > 0 
                ? ((value.attempts - value.failures) / value.attempts * 100).toFixed(2) + '%'
                : 'N/A'
        };
    });
    
    res.json({
        success: true,
        models: stats,
        availableModels: MODEL_PRIORITY.map(m => `${m.name} (${m.provider})`)
    });
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        activeSessions: chatSessions.size,
        geminiConfigured: !!process.env.GEMINI_API_KEY,
        groqConfigured: !!process.env.GROQ_API_KEY,
        fallbackEnabled: true,
        modelCount: MODEL_PRIORITY.length
    });
});

// Test endpoint
app.get('/api/test', (req, res) => {
    const geminiConfigured = !!process.env.GEMINI_API_KEY;
    const groqConfigured = !!process.env.GROQ_API_KEY;
    res.json({
        status: 'Server is running',
        aiProviders: 'Google Gemini + Groq (with fallback)',
        geminiConfigured,
        groqConfigured,
        timestamp: new Date().toISOString(),
        availableModels: MODEL_PRIORITY.map(m => `${m.name} (${m.provider})`),
        endpoints: [
            'POST /api/chat',
            'POST /api/analyze-symptoms',
            'POST /api/summarize-report',
            'POST /api/medicine-info',
            'POST /api/health-tips',
            'POST /api/diet-plan',
            'POST /api/read-prescription',
            'GET /api/model-stats'
        ]
    });
});

// ==================== PROFILE ROUTES ====================
app.post("/about", requireLogin, async (req, res) => {
    try {
        const { name, age, gender, address, height, weight, bloodGroup, critical } = req.body;
        const userId = req.session.user_id;

        if (!userId) {
            return res.status(401).send("Unauthorized: Please log in again");
        }

        const updatedUser = await User.findByIdAndUpdate(
            userId,
            {
                name,
                age: age || null,
                gender: gender || "NA",
                address: address || "",
                height: height || null,
                weight: weight || null,
                bloodGroup: bloodGroup || "",
                critical: critical || "",
            },
            { new: true, runValidators: true }
        );

        if (!updatedUser) {
            return res.status(404).send("User not found");
        }

        req.session.user_name = updatedUser.name;
        req.session.user_email = updatedUser.email;

        console.log("✅ Profile updated successfully for:", updatedUser.email);
        res.redirect("/about");

    } catch (err) {
        console.error("❌ Error updating profile:", err);
        res.status(500).send("Internal Server Error: " + err.message);
    }
});

app.get("/consult", requireLogin, (req, res) => {
    res.render("consult");
});

app.get("/about", requireLogin, async (req, res) => {
    try {
        const userId = req.session.user_id;

        if (!userId) {
            return res.redirect("/login");
        }

        const currentUser = await User.findById(userId);

        if (!currentUser) {
            return res.redirect("/login");
        }

        res.render("about", { currentUser });

    } catch (err) {
        console.error("❌ Error loading profile:", err);
        res.status(500).send("Internal Server Error");
    }
});

// ==================== ERROR HANDLERS ====================

// 404 handler
app.use((req, res) => {
    res.status(404).render('error', {
        title: 'Page Not Found',
        errorName: '404 Not Found',
        errorMessage: 'The page you are looking for does not exist.',
        stack: ''
    });
});

// Error handler
app.use((err, req, res, next) => {
    console.error('❌ Server Error:', err.message);

    const statusCode = err.statusCode || 500;
    const message = err.isOperational ? err.message : 'Something went wrong on the server.';

    res.status(statusCode).render('error', {
        title: statusCode === 404 ? 'Page Not Found' : 'Error',
        errorName: err.name || 'Error',
        errorMessage: message,
        stack: process.env.NODE_ENV === 'development' ? err.stack : ''
    });
});

// ==================== START SERVER ====================
const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
    console.log('\n' + '='.repeat(60));
    console.log('🏥 HEALTHCARE AI SERVER STARTED');
    console.log('='.repeat(60));
    console.log(`🚀 Server URL: http://localhost:${PORT}`);
    console.log(`🔑 Gemini API:  ${process.env.GEMINI_API_KEY ? '✓ Configured' : '✗ Missing'}`);
    console.log(`🔑 Groq API:    ${process.env.GROQ_API_KEY ? '✓ Configured' : '✗ Missing'}`);
    console.log('🤖 AI Providers: Google Gemini + Groq (Multi-Model Fallback)');
    console.log('📋 Fallback Order:');
    MODEL_PRIORITY.forEach((m, i) => {
        console.log(`   ${i + 1}. ${m.name} (${m.provider})`);
    });
    console.log('='.repeat(60) + '\n');
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\n👋 Shutting down server...');
    console.log('\n📊 Final Model Stats:');
    modelStats.forEach((stats, model) => {
        console.log(`   ${model}: ${stats.attempts} attempts, ${stats.failures} failures, ${stats.rateLimitHits} rate limits`);
    });
    process.exit(0);
});