const mongoose = require("mongoose");

const doctorSchema = new mongoose.Schema({
    name: {
        type: String,
        required: true,
    },
    email: {
        type: String,
        required: true,
        unique: true,
    },
    password: {
        type: String,
        required: true,
    },
    specialization: {
        type: String,
        required: true,
    },
    // New Profile Fields
    phone: String,
    years_experience: Number,
    degrees: String,
    medical_college: String,
    city: String,
    state: String,
    available_time: String,
    
    // Verification Fields
    isVerified: {
        type: Boolean,
        default: false
    },
    verificationRequested: {
        type: Boolean,
        default: false
    },
    
    appointments: [{
        patientId: {
            type: mongoose.Schema.Types.ObjectId,
            ref: "User"
        },
        patientName: String,
        // Added Form Data Fields
        age: Number,
        gender: String,
        contact: String,
        symptoms: String,
        bloodGroup: String,
        chronicConditions: String,
        address: {
            type: String,
            default: 'N/A'
        }, 
        
        date: Date,
        reason: String,
        status: {
            type: String,
            enum: ['Pending', 'Completed', 'Cancelled'],
            default: 'Pending'
        }
    }]
});

module.exports = mongoose.model("Doctor", doctorSchema);
