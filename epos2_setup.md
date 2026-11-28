# EPOS2 startup wizard setup parameters

Recorded from EPOS Studio Startup Wizard for **EPOS2 [Node 24]** (`maxon EC Motor` bench setup).

## Configuration summary

| Parameter | Value | Unit |
| --- | --- | --- |
| Node ID | `24` | — |
| CANopen SDO request / response ID | `0x618` / `0x598` | — |
| Communication interface | `USB` (`USB0`) | — |
| Transfer rate | `1000000` | bps |
| Auxiliary regulation | `None` | — |
| Motor type | `maxon EC Motor` | — |
| Commutation type | `Sinus (Incremental Encoder 1 + Hallsensor)` | — |
| Main sensor type | `Incremental Encoder 1 w/o index (2ch)` | — |
| Max. application speed | `1600` | rpm |
| Nominal current | `5000` | mA |
| Max. output current limit | `10000` | mA |
| Thermal time constant winding | `34.2` | s |
| Number of pole pairs | `11` | — |
| Encoder resolution | `6400` | pulse/turn |
| Position resolution | `25600` | qc/turn |
| Inverted encoder counting direction | `FALSE` | — |
| Max. following error | `2000` | qc |

## Wizard steps

### Step 2: Communication setting

- **Interface:** `USB`
- **Port:** `USB0`
- **Transfer Rate:** `1000000` bps
- Screenshot: [wizard_step2_communication.png](docs/wizard_step2_communication.png)

### Step 3: Auxiliary regulation

- **Auxiliary Regulation:** `None`
- Screenshot: [wizard_step3_aux_regulation.png](docs/wizard_step3_aux_regulation.png)

### Step 4: Motor type

- **Motor Type:** `maxon EC Motor`
- Screenshot: [wizard_step4_motor_type.png](docs/wizard_step4_motor_type.png)

### Step 5: EC motor commutation type

- **Commutation Type:** `Sinus (Incremental Encoder 1 + Hallsensor)`
- Screenshot: [wizard_step5_commutation.png](docs/wizard_step5_commutation.png)

### Step 6: Main sensor type

- **Main Sensor Type:** `Incremental Encoder 1 w/o index (2ch)`
- Screenshot: [wizard_step6_main_sensor.png](docs/wizard_step6_main_sensor.png)

### Step 7: System data

- **Max. Application Speed:** `1600` rpm
- **Nominal Current:** `5000` mA
- **Max. Output Current Limit:** `10000` mA
- **Thermal Time Constant Winding:** `34.2` s
- **Number of Pole Pairs:** `11`
- Screenshot: [wizard_step7_system_data.png](docs/wizard_step7_system_data.png)

### Step 8: Incremental encoder 1 w/o index (2ch)

- **Encoder Resolution:** `6400` pulse/turn
- **Position Resolution:** `25600` qc/turn (`4 * 6400`)
- **Inverted Encoder Counting Direction:** `FALSE`
- Screenshot: [wizard_step8_encoder.png](docs/wizard_step8_encoder.png)

### Step 9: Safety parameter position

- **Max. Following Error:** `2000` qc
- Screenshot: [wizard_step9_following_error.png](docs/wizard_step9_following_error.png)

### Step 10: Configuration summary

- **Communication:** `USB - USB0`
- **Protocol Setting:** `1000000 bps, Node 24`
- **Motor Type:** `EC Motor`
- **Commutation:** `Sinus (Incremental Encoder 1 + Hallsensor)`
- **Main Sensor:** `Incremental Encoder 1 w/o index (2ch)`
- **Resolution:** `25600 qc/turn`
- Screenshot: [wizard_step10_summary.png](docs/wizard_step10_summary.png)
