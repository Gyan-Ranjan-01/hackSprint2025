# 🩺 DocOnCall

<p align="center">
  <b>AI-Powered Healthcare Assistance Platform</b><br>
  Built with Node.js, Express.js, MongoDB, Gemini AI & EJS
</p>

![Node.js](https://img.shields.io/badge/Node.js-Backend-green)
![MongoDB](https://img.shields.io/badge/MongoDB-Database-green)
![Express.js](https://img.shields.io/badge/Express.js-Framework-black)
![Gemini AI](https://img.shields.io/badge/AI-Gemini-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

An AI-powered healthcare assistance platform that helps users analyze symptoms, consult doctors, receive medical guidance, and manage healthcare interactions through an intuitive web interface.

🔗 Live Demo: https://doconcall-by-devai.onrender.com/

---

## 🚀 Features

### 👤 User Features
- Secure user authentication
- Symptom analysis system
- AI-powered medical assistance
- Doctor consultation booking
- Medical information access
- Health tips and diet plans
- Chatbot support

### 🩺 Doctor Features
- Dedicated doctor login/signup
- Doctor dashboard
- Consultation management
- Patient interaction interface

### 🤖 AI Integration
- Gemini AI integration
- Groq API support
- Intelligent symptom analysis
- AI-generated healthcare guidance

### 📧 Utility Features
- Email notifications using Nodemailer
- Session management
- Scheduled background tasks using Cron
- Error handling pages

---

## ⚡ Workflow

1. User registers/login securely
2. User enters symptoms or healthcare query
3. AI analyzes the input using Gemini/Groq APIs
4. System generates healthcare guidance and recommendations
5. User can access additional healthcare tools and reports
6. Doctors can manage consultations through the dashboard

---

## 🛠️ Tech Stack

### Frontend
- EJS
- HTML5
- CSS3
- JavaScript

### Backend
- Node.js
- Express.js

### Database
- MongoDB
- Mongoose

### Authentication
- Passport.js
- Express Sessions
- bcrypt

### AI & APIs
- Google Gemini API
- Groq SDK

### Other Tools
- Nodemailer
- Node-cron
- Method-override

---

## 📂 Project Structure

```bash
DocOnCall/
│── models/
│   ├── doctor.js
│   └── user.js
│
│── views/
│   ├── partials/
│   ├── doctor-dashboard.ejs
│   ├── symptom-analysis.ejs
│   ├── consult.ejs
│   └── ...
│
│── app.js
│── package.json
│── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Gyan-Ranjan-01/DocOnCall.git
```

### 2️⃣ Move into project directory

```bash
cd DocOnCall
```

### 3️⃣ Install dependencies

```bash
npm install
```

### 4️⃣ Configure environment variables

Create a `.env` file in the root directory and add:

```env
MONGO_URI=your_mongodb_connection
SESSION_SECRET=your_secret
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
EMAIL=your_email
EMAIL_PASSWORD=your_password
```

### 5️⃣ Run the project

```bash
node app.js
```

---

## 🌐 Live Preview

👉 https://doconcall-by-devai.onrender.com/

---

## 📸 Project Screenshots

### 🏠 Homepage
<img src="assets/Screenshot (1477).png" width="800"/>

---

### 🔐 User Authentication
<img src="assets/Screenshot (1478).png" width="800"/>

---

### 👨‍⚕️ Healthcare Dashboard
<img src="assets/Screenshot (1479).png" width="800"/>

---

### 🩺 AI Healthcare Features
<img src="assets/Screenshot (1480).png" width="800"/>

---

### 🤖 AI Chatbot
<img src="assets/Screenshot (1481).png" width="800"/>

---

### 📋 Medical Report Summary
<img src="assets/Screenshot (1482).png" width="800"/>

---

## 🧠 What I Learned

- Full-stack application architecture
- Authentication and session handling
- MongoDB schema design
- AI API integration
- Dynamic rendering with EJS
- Backend routing and middleware handling
- Real-world healthcare workflow implementation

---

## 🔮 Future Improvements

- Video consultation integration
- Real-time chat using WebSockets
- Appointment scheduling system
- Medical report upload
- AI-based prescription assistance
- Deployment optimization
- JWT-based authentication
- REST API modularization
- Docker containerization

---

## 👨‍💻 Team

Developed by **DevStormers**

### Contributors
- Gyan Ranjan
- Team Devstormers from HackSprint 2025

---

## 📬 Contact

### Gyan Ranjan

🔗 LinkedIn: https://www.linkedin.com/in/gyan-ranjan-/

🔗 GitHub: https://github.com/Gyan-Ranjan-01

---

## ⭐ If you found this project useful, give it a star!
