"""
ECU Knowledge Base
Used to validate if student responses mention correct ECU names
"""

# BMW ECU abbreviation list
BMW_ECU_LIST = {
    # --- 1. Powertrain & Chassis ---
    "DME", "DDE", "EGS", "VTG", "GWS",
    "DSC", "DSCi", "EPS", "AL", "HSR",
    "EHC", "VDP", "EDC", "EARS",
    "GHAS", "EMF", "PCU",
    
    # --- 2. Body & Gateway ---
    "CAS", "FEM", "BDC", "BDC_BODY", "BDC_GW",
    "ZGM", "JBE", "REM", "FRM",
    "BCP", "VIP",
    "FLM", "FLM2", "STML", "STMR",
    "HKL", "FZD", "IHKA", "IHKA_PRO",
    "SMFA", "SMBF", "SMFAH", "SMBFH",
    
    # --- 3. Infotainment & Cockpit ---
    "HU_H", "HU_NBT", "HU_ENTRY", "HU_CIC",
    "MGU", "MGU21", "MGU22",
    "IDC", "IDC23",
    "RAM", "BOOSTER",
    "KOMBI", "DKOMBI", "DKOMBI4", "DKOMBI8",
    "ATM", "TCB", "WIB",
    "CON", "ZBE", "TBX",
    
    # --- 4. ADAS & Safety ---
    "ACSM", "MRS",
    "ICM", "SAS", "SAS2", "SAS3",
    "KAFAS", "KAFAS2", "KAFAS4",
    "TRSVC", "ICAM", "ADCAM",
    "PMA", "PAD",
    "LRR", "SRR", "RSU",
    
    # --- 5. EV & High Voltage ---
    "EME", "SME", "HVS", "CSC",
    "CCU", "KLE", "LIM",
    "EWS", "ZKE", "GM",
}

# ECU abbreviation -> related English keywords
BMW_ECU_KEYWORDS = {
    # Powertrain & Chassis
    "DME": ["Digital Motor Electronics", "engine control", "motor electronics"],
    "DDE": ["Digital Diesel Electronics", "diesel control"],
    "EGS": ["Electronic Transmission Control", "gearbox", "transmission"],
    "VTG": ["Transfer Case", "transfer box"],
    "GWS": ["Gear Selector Switch", "gear selector"],
    "DSC": ["Dynamic Stability Control", "stability control", "braking control"],
    "DSCi": ["Dynamic Stability Control integrated", "iBrake"],
    "EPS": ["Electric Power Steering", "power steering"],
    "AL": ["Active Steering"],
    "HSR": ["Rear Axle Steering", "rear wheel steering"],
    "EHC": ["Electronic Height Control", "air suspension"],
    "VDP": ["Vertical Dynamics Platform"],
    "EDC": ["Electronic Damper Control", "damper control"],
    "EARS": ["Active Roll Stabilization", "anti-roll"],
    "GHAS": ["Rear Axle Differential Lock", "differential lock"],
    "EMF": ["Electronic Parking Brake", "parking brake"],
    "PCU": ["Power Control Unit"],
    
    # Body & Gateway
    "CAS": ["Car Access System", "access system", "immobilizer"],
    "FEM": ["Front Electronic Module", "front module"],
    "BDC": ["Body Domain Controller", "body controller"],
    "ZGM": ["Central Gateway Module", "gateway"],
    "JBE": ["Junction Box Electronics", "junction box"],
    "REM": ["Rear Electronic Module", "rear module"],
    "FRM": ["Footwell Module", "footwell"],
    "BCP": ["Basic Central Platform", "central platform"],
    "VIP": ["Vehicle Integration Platform"],
    "FLM": ["Front Light Module", "headlight module"],
    "STML": ["Headlight Driver Left"],
    "STMR": ["Headlight Driver Right"],
    "HKL": ["Tailgate Lift", "tailgate"],
    "FZD": ["Roof Function Center", "roof module"],
    "IHKA": ["Climate Control", "air conditioning", "HVAC"],
    "SMFA": ["Seat Module Front Driver", "driver seat"],
    "SMBF": ["Seat Module Front Passenger", "passenger seat"],
    
    # Infotainment & Cockpit
    "HU_H": ["Head Unit High", "infotainment"],
    "HU_NBT": ["Head Unit NBT", "navigation"],
    "HU_CIC": ["Car Information Computer"],
    "MGU": ["Media Graphic Unit", "iDrive", "media unit"],
    "IDC": ["Info Domain Computer", "iDrive 9"],
    "RAM": ["Receiver Audio Module", "audio module"],
    "BOOSTER": ["Audio Amplifier", "amplifier"],
    "KOMBI": ["Instrument Cluster", "cluster", "dashboard"],
    "DKOMBI": ["Digital Instrument Cluster"],
    "ATM": ["Telematics", "T-Box"],
    "TCB": ["Telematics Communication Box"],
    "WIB": ["Wireless Interface Box", "wireless"],
    "CON": ["Controller", "iDrive controller"],
    "ZBE": ["Central Control Unit", "touchpad"],
    
    # ADAS & Safety
    "ACSM": ["Crash Safety Module", "airbag", "safety module", "crash safety"],
    "MRS": ["Multiple Restraint System", "restraint system", "airbag control"],
    "ICM": ["Integrated Chassis Management"],
    "SAS": ["Driver Assistance System", "ADAS"],
    "KAFAS": ["Camera Assist System", "front camera"],
    "TRSVC": ["Surround View Camera", "parking camera"],
    "ICAM": ["Interior Camera"],
    "ADCAM": ["Advanced Camera"],
    "PMA": ["Park Distance Control", "parking assist", "PDC"],
    "PAD": ["Parking Assistance", "auto parking"],
    "LRR": ["Long Range Radar", "radar"],
    "SRR": ["Short Range Radar"],
    "RSU": ["Radar Sensor Unit"],
    
    # EV & High Voltage
    "EME": ["Electric Motor Electronics", "electric motor", "e-motor"],
    "SME": ["Battery Management", "high voltage battery"],
    "HVS": ["High Voltage Safety", "HV safety"],
    "CSC": ["Cell Supervision Circuit", "cell monitor"],
    "CCU": ["Combined Charging Unit", "charging unit"],
    "KLE": ["Convenience Charging Electronics", "charging"],
    "LIM": ["Charging Interface Module"],
    
    # Legacy
    "EWS": ["Electronic Immobilizer", "immobilizer"],
    "ZKE": ["Central Body Electronics"],
    "GM": ["General Module"],
}


def get_ecu_abbrevs_from_name(ecu_name: str) -> list:
    """
    Extract abbreviations from ECU full name.
    Example: "Crash Safety Module (ACSM/SIM)" -> ["ACSM", "SIM"]
    """
    import re
    abbrevs = []
    
    # Find abbreviations in parentheses
    matches = re.findall(r'\(([A-Z0-9/_]+)\)', ecu_name)
    for match in matches:
        abbrevs.extend(match.split('/'))
    
    # If no parentheses, try to match all-uppercase abbreviations
    if not abbrevs:
        matches = re.findall(r'\b([A-Z]{2,}[0-9]*)\b', ecu_name)
        abbrevs.extend(matches)
    
    return abbrevs


def validate_ecu_in_response(response: str, correct_ecu_name: str) -> float:
    """
    Validate if student response mentions correct ECU.
    
    Args:
        response: Student's response text
        correct_ecu_name: Correct ECU name (from ecuid_name.json query)
    
    Returns:
        Match score (0.0 - 1.0)
    """
    import re
    
    if not response or not correct_ecu_name:
        return 0.5  # Cannot validate, return neutral score
    
    response_upper = response.upper()
    
    # 1. Extract correct ECU abbreviations
    correct_abbrevs = get_ecu_abbrevs_from_name(correct_ecu_name)
    
    # Short abbreviations need full word match to avoid false matches
    SHORT_ABBREVS = {"AL", "GM", "CON", "VIP", "RAM", "EDC", "EPS", "HKL", "FZD", "SAS"}
    
    def is_word_match(abbrev: str, text: str) -> bool:
        """Check for full word match (avoid substring false matches)"""
        if len(abbrev) <= 3 or abbrev in SHORT_ABBREVS:
            # Short abbreviations need full word match
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            return bool(re.search(pattern, text, re.IGNORECASE))
        else:
            # Long abbreviations can use simple contains
            return abbrev.upper() in text.upper()
    
    # 2. Exact match ECU abbreviation
    for abbrev in correct_abbrevs:
        if is_word_match(abbrev, response):
            return 1.0  # Full match
    
    # 3. Keyword match
    for abbrev in correct_abbrevs:
        if abbrev in BMW_ECU_KEYWORDS:
            for keyword in BMW_ECU_KEYWORDS[abbrev]:
                if keyword.upper() in response_upper:
                    return 0.7  # Keyword match
    
    # 4. Check if wrong ECU is mentioned
    # Only check abbreviations with length >= 4 to avoid false matches
    correct_abbrevs_upper = {a.upper() for a in correct_abbrevs}
    SAFE_ECU_LIST = {ecu for ecu in BMW_ECU_LIST if len(ecu) >= 4 and ecu not in SHORT_ABBREVS}
    
    for ecu in SAFE_ECU_LIST:
        if is_word_match(ecu, response) and ecu.upper() not in correct_abbrevs_upper:
            return 0.2  # Wrong ECU mentioned -> penalty
    
    # 5. No ECU mentioned -> slight penalty
    return 0.4
