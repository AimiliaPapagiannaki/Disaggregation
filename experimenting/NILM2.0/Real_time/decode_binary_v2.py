##
# @namespace decode_binary_v2
# @brief Mosquitto message decoder for v2 messages

import struct

### @private
PHASE_SIZE = 53 
### @private
HARM_SIZE = 21
### @private
HARM_WIN = 4
### @private
CH_WIN_SIZE = 4

### @private
has_crc = True
### @private
logger = None

### Calculate the CRC of the given data
def calculate_crc(data):  
  crc = 0
  for b in data:
    crc = crc ^ b
  return crc

##
# @brief Test the CRC of the received payload against the received CRC
# @param payload The payload. The last byte if the received CRC
# @returns True if the CRC matches, False otherwise
def test_crc(payload):
    received_crc = payload[-1]
    calc_crc = calculate_crc(payload[0:-1] )
    if calc_crc != received_crc:
            logger.error("CRC mismatch")
            return False
    return True

##
# @brief Get the timestamp from a stamped message
#
# @returns a tuple (ts, real_payload)\n
# <b>Ts:</b> the timestamp (millisecond since 1970)\n
# <b>real_payload:</b> the rest of the payload
def get_ts(payload):
    bstart = 0
    bend = bstart + 6
    [sec, ms] = struct.unpack("<LH", payload[bstart:bend])
        
    ts = sec * 1000 +ms
    real_payload = payload[bend:]
    return ts, real_payload
  
##
# @brief Decode a measurement message. 
# @param payload The payload after <i>get_ts</i>
# @return A measurement dictionary or <i>None</i> on error.
def on_measurement(payload, show_extra_vals):    
    bstart = 0
    #1 + 4 + 4 = 9
    bend = bstart + 9
    [ptype, sn, time] = struct.unpack("<Bii", payload[bstart:bend])
        
    real_payload = payload[bend:-1]
    
    data = None
    if ptype == 0:
        data = decodePhases(real_payload, show_extra_vals)
    elif ptype == 1:
        data = decodeHarmonics(real_payload)
    elif ptype == 2:
        data = decodeChanges(real_payload)    
    
    if data is not None:
        data['sN'] = sn

    return data

### @private
def decodeHarmonics(payload):
    plen = len(payload)
    num_of_phases = int(plen / HARM_SIZE)
    
    bstart = 0
    bend = 0
    harm = {}
    
    for i in range(num_of_phases):        
        bstart = bend
        bend = bstart + 1
        ph = struct.unpack("<B", payload[bstart:bend])[0]
        phc = None
        if ph == 1:
            phc = 'A'
        elif ph == 2:
            phc = 'B'
        elif ph == 3:
            phc = 'C'
        else:
            return None
        
        for i in range(HARM_WIN):
            bstart = bend
            bend = bstart + 5
            [hc, vhd, ihd]  = struct.unpack("<BHH", payload[bstart:bend])
            
            if hc == 1:
              vhdStr = 'vthd' + phc
              ihdStr = 'ithd' + phc
            else:
              hcc=str(hc).zfill(2)    
              vhdStr = 'vhd' + hcc + phc
              ihdStr = 'ihd' + hcc + phc
            
            harm[vhdStr] = vhd / 100
            harm[ihdStr] = ihd / 100
    return harm

### @private
def decodePhases(payload, show_extra_vals):
    phases = {}
    has_iTHD = False
    has_iN = False
    
    plen = len(payload)
    num_of_phases_fl = (plen) / PHASE_SIZE
    num_of_phases = int(num_of_phases_fl)

    if num_of_phases_fl == num_of_phases:
        has_iN = False
        has_iTHD = False
    else:
        num_of_phases_fl = plen / (PHASE_SIZE + 4)
        num_of_phases = int(num_of_phases_fl)
        if num_of_phases_fl == num_of_phases:
            has_iN = True
            has_iTHD = False
        else:
            has_iN = True
            has_iTHD = True
  
    ithd = 0
    bstart = 0
    bend = bstart
    
    
    for i in range(num_of_phases):
        L = {}
        
        fvals = [] 
        ivals = [] 
        phase = {}
        
        bstart = bend
        bend = bstart + 5 # 1 + 4 
        
        [ph, curr] = struct.unpack("<Bf", payload[bstart:bend])
        
        if ph == 1:
            phc = 'A'
        elif ph == 2:
            phc = 'B'
        elif ph == 3:
            phc = 'C'
        else:            
            return None
             
        if has_iN:
          bstart = bend
          bend = bstart + 4
          iN = struct.unpack("<f", payload[bstart:bend])[0]
        
        bstart = bend
        bend = bstart + (4 * 8)
        fvals = struct.unpack("<ffffffff", payload[bstart:bend])
                
        if has_iTHD and ph == 1:
            bstart = bend
            bend = bstart + 2
            ithd = struct.unpack("<H", payload[bstart:bend])[0]
            ithd = ithd / 100          
        
        bstart = bend
        bend = bstart + (4 * 4)
        ivals = struct.unpack("<IIII", payload[bstart:bend])

        """
        phases['vlt' + phc]=str(fvals[0])        
        phases['cur' + phc]=str(curr)
        phases['iN'] = iN_str
        
        phases['crnrg' + phc]=str(ivals[2])
        phases['prnrg' + phc]=str(ivals[3])
        phases['efrq' + phc]=str(fvals[4])
        """
        
        """
        phases['apwr' + phc]=str(fvals[3])        
        phases['cos' + phc]=str(fvals[5])
        phases['scre' + phc]=str(fvals[6])
        phases['angle' + phc]=str(fvals[7])        
        phases['cnrg' + phc]=str(ivals[0])
        phases['pnrg' + phc]=str(ivals[1])                
        phases['vlt' + phc]=str(fvals[0])
        """
                
        phases['cur' + phc]=curr
        phases['vlt' + phc]=fvals[0]
        phases['pwr' + phc]=fvals[1]       
        #phases['rpwr' + phc]=fvals[2]
        phases['apwr' + phc]=fvals[3]      
        phases['cos' + phc]=fvals[5]
        phases['scre' + phc]=fvals[6]
        phases['angle' + phc]=fvals[7]
        phases['cnrg' + phc]=ivals[0]
        phases['pnrg' + phc]=ivals[1]  
        
        if show_extra_vals:
            phases['rpwr' + phc]=fvals[2] 
        
    if has_iTHD:
        phases['ithd'] = ithd
    return phases
  
### @private
def decodeChanges(payload):
    ch = {}
    bstart = 0
    bend = bstart + 19 #1 + 1 + 4 + 4 + 4 + 4 + 1
    [phase, s0, utc, msT, avgSP, avgSR, sign ] = struct.unpack("<bbiIffb", payload[bstart:bend])
    
    if s0 != 0:
        return None
      
    if phase == 0:
        phc = 'A'
    elif phase == 1:
        phc = 'B'
    elif phase == 2:
        phc = 'C'
    else:
        return None

    for i in range(CH_WIN_SIZE):
        bstart = bend
        bend = bstart + 1 + (8 * 4)
        
        vals = struct.unpack("<bffffffff", payload[bstart:bend]) 
        
        si = vals[0]
        if si < 1 or si > CH_WIN_SIZE:
            return None
        
        ch['avgP' + phc + '_' + str(si) ]=vals[1]
        ch['minP' + phc + '_' + str(si) ]=vals[2]
        ch['maxP' + phc + '_' + str(si) ]=vals[3]
        ch['stdP' + phc + '_' + str(si) ]=vals[4]
        ch['avgR' + phc + '_' + str(si) ]=vals[5]
        ch['minR' + phc + '_' + str(si) ]=vals[6]
        ch['maxR' + phc + '_' + str(si) ]=vals[7]
        ch['stdR' + phc + '_' + str(si) ]=vals[8]
    

    ch['utc_' + phc] = utc
    ch['msT_' + phc] = msT
    ch['avgSP_'+ phc ] = avgSP
    ch['avgSR_' + phc ] = avgSR
    ch['sign_' + phc] = sign
    
    return ch
    
### @private
def decodeInfo(payload, bstart):
  ret = {}
  data = payload[bstart:]
  dataStr = data.decode("ascii")  
  newLine = dataStr.find('\n')
  
  strs = dataStr.split('\n')
  version = strs[0]
  resetReson = strs[1] 
  
  if len(strs) > 2:
    resetReson = resetReson + ', ' + strs[2]
  
  ret['Version'] = version
  ret['Reset_reason'] = resetReson
  
  return ret
  
##
# @brief Decodes an attribute or info message
# @param message The message payload.\n
# In case of attribute, this is exactly the received message.\n
# In case of info, this is the payload after <i>get_ts</i>
# @returns a dictionary with the message values
def on_attribute(message):
    payload_str = message.decode()
    lines = payload_str.splitlines()
    payload_dict={}
    for line in lines:
        vals = line.split(':',1)
        if len(vals) > 1:
          if len(vals[0]) > 0 and len(vals[1]) > 0:
            payload_dict[vals[0]]=vals[1]
          elif len(vals[0]) > 0:
            payload_dict[vals[0]]=''
        elif len(vals) > 0:
            payload_dict[vals[0]]=''
    return payload_dict
