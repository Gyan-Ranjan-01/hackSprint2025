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
    secret: "thisshouldbeabettersecret!",
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

// ==================== GOOGLE GEMINI AI INITIALIZATION ====================
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const chatSessions = new Map();

console.log('🤖 Initializing Google Gemini AI...');
console.log(`🔑 API Key Status: ${process.env.GEMINI_API_KEY ? 'Configured ✓' : 'Missing ✗'}`);

// ==================== MONGODB CONNECTION ====================
async function main() {
    console.log("mongodb://127.0.0.1:27017/hacksprint");
    await mongoose.connect("mongodb://127.0.0.1:27017/hacksprint");
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
    
    // Validation
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

    // Successful login
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
    
    // Validation
    if (!name || !email || !password) {
        throw new AppError("All fields are required", 400);
    }
    
    // Check if user already exists
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
    
    const model = genAI.getGenerativeModel({ 
        model: "gemini-2.0-flash",
        generationConfig: {
            temperature: 0.7,
            topK: 40,
            topP: 0.95,
            maxOutputTokens: 300,
        }
    });
    
    let chat;
    if (!chatSessions.has(sessionId)) {
        chat = model.startChat({
            history: [
                {
                    role: "user",
                    parts: [{ text: "You are Dr. AI, a friendly and empathetic medical assistant chatbot for a virtual healthcare platform. Your role is to: 1) Ask relevant questions about symptoms, 2) Provide general health guidance, 3) Show empathy and be reassuring, 4) Keep responses concise (2-4 sentences), 5) ALWAYS remind users this is not a replacement for professional medical advice. Be warm, professional, and helpful." }]
                },
                {
                    role: "model",
                    parts: [{ text: "Hello! I'm Dr. AI, your virtual health assistant. I'm here to help answer your health questions and provide general guidance. Please remember that I'm not a replacement for professional medical advice. How can I assist you today?" }]
                }
            ]
        });
        chatSessions.set(sessionId, chat);
    } else {
        chat = chatSessions.get(sessionId);
    }
    
    const result = await chat.sendMessage(message);
    const reply = result.response.text();
    
    console.log(`🤖 Reply: ${reply.substring(0, 100)}...`);
    
    res.json({ 
        reply,
        success: true,
        sessionId 
    });
}));

// 2. Symptom Analysis
app.post('/api/analyze-symptoms', requireLogin, wrapAsync(async (req, res) => {
    const { symptoms, age, gender, duration } = req.body;
    
    if (!symptoms) {
        throw new AppError('Symptoms are required', 400);
    }
    
    console.log('🔍 Analyzing symptoms:', symptoms);
    
    const model = genAI.getGenerativeModel({ 
        model: "gemini-2.0-flash",
        generationConfig: {
            temperature: 0.3,
            maxOutputTokens: 600,
        }
    });
    
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

    const result = await model.generateContent(prompt);
    const analysis = result.response.text();
    
    console.log('✅ Analysis complete');
    
    res.json({ 
        analysis,
        success: true 
    });
}));

// 3. Medical Report Summary
app.post('/api/summarize-report', requireLogin, wrapAsync(async (req, res) => {
    const { reportText, reportType } = req.body;
    
    if (!reportText) {
        throw new AppError('Report text is required', 400);
    }
    
    console.log('📄 Summarizing report...');
    
    const model = genAI.getGenerativeModel({ 
        model: "gemini-2.0-flash",
        generationConfig: {
            temperature: 0.4,
            maxOutputTokens: 500,
        }
    });
    
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

    const result = await model.generateContent(prompt);
    const summary = result.response.text();
    
    console.log('✅ Summary generated');
    
    res.json({ 
        summary,
        success: true 
    });
}));

// 4. Medicine Information
app.post('/api/medicine-info', requireLogin, wrapAsync(async (req, res) => {
    const { medicineName } = req.body;
    
    if (!medicineName) {
        throw new AppError('Medicine name is required', 400);
    }
    
    console.log('💊 Getting info for:', medicineName);
    
    const model = genAI.getGenerativeModel({ 
        model: "gemini-2.0-flash",
        generationConfig: {
            temperature: 0.3,
            maxOutputTokens: 400,
        }
    });
    
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

    const result = await model.generateContent(prompt);
    const info = result.response.text();
    
    console.log('✅ Medicine info retrieved');
    
    res.json({ 
        info,
        success: true 
    });
}));

// 5. Health Tips Generator
app.post('/api/health-tips', requireLogin, wrapAsync(async (req, res) => {
    const { category, userProfile } = req.body;
    
    console.log('💡 Generating tips for:', category);
    
    const model = genAI.getGenerativeModel({ 
        model: "gemini-2.0-flash",
        generationConfig: {
            temperature: 0.7,
            maxOutputTokens: 500,
        }
    });
    
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

    const result = await model.generateContent(prompt);
    const tips = result.response.text();
    
    console.log('✅ Tips generated');
    
    res.json({ 
        tips,
        success: true 
    });
}));

// 6. Diet Plan Generator
app.post('/api/diet-plan', requireLogin, wrapAsync(async (req, res) => {
    const { goal, restrictions, preferences } = req.body;
    
    console.log('🥗 Generating diet plan for goal:', goal);
    
    const model = genAI.getGenerativeModel({ 
        model: "gemini-2.0-flash",
        generationConfig: {
            temperature: 0.6,
            maxOutputTokens: 800,
        }
    });
    
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

    const result = await model.generateContent(prompt);
    const plan = result.response.text();
    
    console.log('✅ Diet plan generated');
    
    res.json({ 
        plan,
        success: true 
    });
}));

// 7. Prescription Image Analysis
app.post('/api/read-prescription', requireLogin, wrapAsync(async (req, res) => {
    const { imageBase64 } = req.body;
    
    if (!imageBase64) {
        throw new AppError('Image data is required', 400);
    }
    
    console.log('📸 Analyzing prescription image...');
    
    const model = genAI.getGenerativeModel({ 
        model: "gemini-2.0-flash",
        generationConfig: {
            temperature: 0.2,
            maxOutputTokens: 500,
        }
    });
    
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

    const result = await model.generateContent([prompt, ...imageParts]);
    const analysis = result.response.text();
    
    console.log('✅ Prescription analyzed');
    
    res.json({ 
        analysis,
        success: true 
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

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        activeSessions: chatSessions.size,
        geminiConfigured: !!process.env.GEMINI_API_KEY
    });
});

// Test endpoint
app.get('/api/test', (req, res) => {
    const isConfigured = !!process.env.GEMINI_API_KEY;
    res.json({ 
        status: 'Server is running',
        aiProvider: 'Google Gemini',
        geminiConfigured: isConfigured,
        timestamp: new Date().toISOString(),
        endpoints: [
            'POST /api/chat',
            'POST /api/analyze-symptoms',
            'POST /api/summarize-report',
            'POST /api/medicine-info',
            'POST /api/health-tips',
            'POST /api/diet-plan',
            'POST /api/read-prescription'
        ]
    });
});


// ==================== ROUTE: POST /about ====================
app.post("/about", requireLogin, async (req, res) => {
  try {
    const { name, age, gender, address, height, weight, bloodGroup, critical } = req.body;

    // ✅ Always use the same session key
    const userId = req.session.user_id;

    if (!userId) {
      return res.status(401).send("Unauthorized: Please log in again");
    }

    // ✅ Update user info
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

    // ✅ Optionally update name/email in session for future use
    req.session.user_name = updatedUser.name;
    req.session.user_email = updatedUser.email;

    console.log("✅ Profile updated successfully for:", updatedUser.email);

    // ✅ Redirect to about page after saving
    res.redirect("/about");

  } catch (err) {
    console.error("❌ Error updating profile:", err);
    res.status(500).send("Internal Server Error: " + err.message);
  }
});

// consultation with doctor route
app.get("/consult", requireLogin, (req, res) => {
    res.render("consult");
});

// ==================== ROUTE: GET /about ====================
app.get("/about", requireLogin, async (req, res) => {
  try {
    // ✅ Use same session key
    const userId = req.session.user_id;

    if (!userId) {
      return res.redirect("/login");
    }

    // ✅ Fetch the user from DB
    const currentUser = await User.findById(userId);

    if (!currentUser) {
      return res.redirect("/login");
    }

    // ✅ Render the about page with currentUser
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
    console.error('❌ Server Error');

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
app.listen(8080, () => {
    console.log('\n' + '='.repeat(50));
    console.log('🏥 HEALTHCARE AI SERVER STARTED');
    console.log('='.repeat(50));
    console.log('🚀 Server URL: http://localhost:8080');
    console.log(`🔑 API Key:    ${process.env.GEMINI_API_KEY ? '✓ Configured' : '✗ Missing'}`);
    console.log('🤖 AI Provider: Google Gemini');
    console.log('='.repeat(50) + '\n');
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\n👋 Shutting down server...');
    process.exit(0);
});