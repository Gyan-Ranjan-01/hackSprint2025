const mongoose = require('mongoose');
const Schema = mongoose.Schema;
const passportLocalMongoose = require('passport-local-mongoose');

const UserSchema = new Schema({
    name: {
        type: String,
        required: true,
        
    },

    email: {
        type: String,
        required: true,
        unique: true
    },
    password: {
        type: String,
        required: true
    },
    age: { type: Number, default: null },
    gender: { type: String, enum: ["Male", "Female", "Other",'NA'], default: "NA" },
    address: { type: String, default: "" },
    height: { type: Number, default: null },
    weight: { type: Number, default: null },
    bloodGroup: { type: String, default: "" },
    critical: { type: String, default: "" },

  createdAt: { type: Date, default: Date.now }
});

UserSchema.plugin(passportLocalMongoose, { usernameField: 'email' });

module.exports = mongoose.model('User', UserSchema);
