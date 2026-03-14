-- database/db_setup.sql
-- ─────────────────────────────────────────────────────────────────
-- Smart Campus Navigation — Database Schema & Seed Data
-- Compatible with both MySQL and SQLite
-- ─────────────────────────────────────────────────────────────────

-- Create the facilities table
CREATE TABLE IF NOT EXISTS facilities (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    category      TEXT    NOT NULL,           -- matches intent: STUDY, FOOD, MEDICAL, ADMIN, LAB, HOSTEL
    description   TEXT,
    latitude      REAL    NOT NULL,
    longitude     REAL    NOT NULL,
    opening_time  TEXT    NOT NULL,           -- format: HH:MM (24h)
    closing_time  TEXT    NOT NULL,           -- format: HH:MM (24h)
    building      TEXT,
    floor         TEXT
);

-- ─── Seed Data ────────────────────────────────────────────────────────────────
-- Using approximate coordinates for a fictional campus
-- Replace with your actual campus coordinates

-- STUDY facilities
INSERT INTO facilities (name, category, description, latitude, longitude, opening_time, closing_time, building, floor) VALUES
('Central Library',          'STUDY', 'Main campus library with silent reading zones and group study rooms', 3.1390, 101.6869, '07:00', '22:00', 'Library Block', 'Ground–4th'),
('Engineering Study Lounge', 'STUDY', 'Dedicated study lounge for engineering students with whiteboards', 3.1395, 101.6875, '08:00', '21:00', 'Engineering Tower', '2nd Floor'),
('24-Hour Study Hall',       'STUDY', 'Always-open study hall with Wi-Fi and power outlets', 3.1382, 101.6860, '00:00', '23:59', 'Student Hub', 'Ground Floor'),
('Postgraduate Reading Room', 'STUDY', 'Quiet reading room exclusively for postgraduate students', 3.1400, 101.6880, '08:00', '20:00', 'Main Block A', '3rd Floor'),
('Open Learning Center',     'STUDY', 'Collaborative space with computers, printers and project rooms', 3.1375, 101.6855, '08:00', '22:00', 'Learning Center', 'Ground Floor');

-- FOOD facilities
INSERT INTO facilities (name, category, description, latitude, longitude, opening_time, closing_time, building, floor) VALUES
('Main Campus Cafeteria',    'FOOD', 'Large food court with multiple cuisines and vegetarian options', 3.1388, 101.6865, '07:00', '21:00', 'Student Center', 'Ground Floor'),
('Engineering Block Canteen','FOOD', 'Quick-service canteen serving rice, noodles and snacks',        3.1398, 101.6878, '07:30', '17:00', 'Engineering Tower', 'Ground Floor'),
('24-Hour Convenience Store','FOOD', 'Round-the-clock snacks, drinks and instant meals',              3.1380, 101.6858, '00:00', '23:59', 'Student Hub', 'Ground Floor'),
('Campus Coffee House',      'FOOD', 'Specialty coffee, sandwiches and pastries in a cozy setting',  3.1393, 101.6872, '08:00', '20:00', 'Arts Block', 'Ground Floor'),
('Hostel Dining Hall',       'FOOD', 'Buffet-style dining for residential students',                 3.1370, 101.6848, '07:00', '22:00', 'Hostel Block A', 'Ground Floor');

-- MEDICAL facilities
INSERT INTO facilities (name, category, description, latitude, longitude, opening_time, closing_time, building, floor) VALUES
('Campus Health Center',     'MEDICAL', 'Full-service student clinic with doctors and nurses on duty',   3.1385, 101.6862, '08:00', '17:00', 'Health Block', 'Ground Floor'),
('Campus Pharmacy',          'MEDICAL', 'Licensed pharmacy stocking prescription and OTC medications',  3.1386, 101.6863, '08:00', '18:00', 'Health Block', 'Ground Floor'),
('Emergency First Aid Post', 'MEDICAL', '24-hour first aid station with trained personnel',             3.1389, 101.6867, '00:00', '23:59', 'Security Office', 'Ground Floor'),
('Mental Health & Counseling','MEDICAL', 'Confidential counseling and psychological support services',  3.1384, 101.6861, '09:00', '17:00', 'Wellness Block', '1st Floor'),
('Dental Clinic',            'MEDICAL', 'Campus dental clinic for students and staff',                  3.1387, 101.6864, '09:00', '16:00', 'Health Block', '1st Floor');

-- ADMIN facilities
INSERT INTO facilities (name, category, description, latitude, longitude, opening_time, closing_time, building, floor) VALUES
('Registrar Office',         'ADMIN', 'Handles course registration, transcripts and academic records',  3.1392, 101.6871, '08:30', '16:30', 'Admin Block', 'Ground Floor'),
('Student Affairs Office',   'ADMIN', 'Student services, clubs, events and welfare enquiries',          3.1391, 101.6870, '08:30', '17:00', 'Admin Block', '1st Floor'),
('Finance & Fees Office',    'ADMIN', 'Fee payments, scholarships and financial aid enquiries',         3.1390, 101.6869, '08:30', '16:00', 'Admin Block', '2nd Floor'),
('Examination Office',       'ADMIN', 'Exam timetables, results, appeals and invigilation matters',     3.1394, 101.6874, '08:30', '16:30', 'Admin Block', '3rd Floor'),
('International Students Office','ADMIN','Visa, immigration and international student support services',3.1396, 101.6876, '09:00', '16:00', 'Admin Block', '2nd Floor');

-- LAB facilities
INSERT INTO facilities (name, category, description, latitude, longitude, opening_time, closing_time, building, floor) VALUES
('Computer Science Lab',     'LAB', 'High-performance workstations, Linux/Windows, coding tools',      3.1397, 101.6877, '08:00', '22:00', 'CS Building', '1st Floor'),
('Electronics & Circuits Lab','LAB', 'Full electronics lab with oscilloscopes and soldering stations', 3.1399, 101.6879, '08:00', '18:00', 'Engineering Tower', '3rd Floor'),
('Chemistry Laboratory',     'LAB', 'Wet chemistry lab for experiments and research',                  3.1401, 101.6881, '08:00', '17:00', 'Science Block', '1st Floor'),
('AI & Machine Learning Lab','LAB', 'GPU workstations for deep learning and AI research',              3.1402, 101.6882, '08:00', '22:00', 'CS Building', '4th Floor'),
('Robotics & IoT Lab',       'LAB', 'Prototyping lab with 3D printers, Arduino kits and robot arms',  3.1403, 101.6883, '09:00', '18:00', 'Engineering Tower', '4th Floor');

-- HOSTEL facilities
INSERT INTO facilities (name, category, description, latitude, longitude, opening_time, closing_time, building, floor) VALUES
('Hostel Block A (Male)',     'HOSTEL', 'Male student dormitory – Rooms, laundry, common room',         3.1371, 101.6849, '00:00', '23:59', 'Hostel Block A', 'All Floors'),
('Hostel Block B (Female)',   'HOSTEL', 'Female student dormitory – Rooms, laundry, study lounge',     3.1369, 101.6847, '00:00', '23:59', 'Hostel Block B', 'All Floors'),
('Hostel Warden Office',      'HOSTEL', 'Warden office for room issues, keys and maintenance requests', 3.1372, 101.6850, '08:00', '22:00', 'Hostel Admin', 'Ground Floor'),
('Hostel Recreation Center',  'HOSTEL', 'Games room, TV lounge and indoor sports area',                3.1368, 101.6846, '10:00', '23:00', 'Hostel Block A', 'Ground Floor'),
('Hostel C (Postgrad)',        'HOSTEL', 'Postgraduate student residence with attached kitchenette',    3.1367, 101.6845, '00:00', '23:59', 'Hostel Block C', 'All Floors');
