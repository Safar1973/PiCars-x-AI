# Importiere notwendige Bibliotheken
import cv2
import numpy as np
import time
from picamera2 import Picamera2
from picarx import Picarx

# --- Initialisiere PiCar-X und Kamera ---
print("Starte Fahrprogramm...")
px = Picarx()
picam2 = Picamera2()

# Setze Kameraaufloesung und Farbformat
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
picam2.preview_configuration.main.size = (CAMERA_WIDTH, CAMERA_HEIGHT)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(2)  # kurze Wartezeit, damit Kamera starten kann

# --- Farb-/Sensor-Konstanten ---
# Farbgrenzen fuer rote Seitenstreifen (HSV-Werte aus Kalibrierung)
lower_red = np.array([110, 80, 200])
upper_red = np.array([179, 255, 255])

# Farbgrenzen fuer schwarzen Streifen
lower_black = np.array([0, 0, 0])
upper_black = np.array([179, 255, 100])

# Mindestflaeche fuer den schwarzen Streifen, um Rauschen zu filtern
BLACK_STRIPE_MIN_AREA = 500 # Diesen Wert eventuell anpassen, je nach Groesse des Streifens

# --- Ultraschallsensor Konstanten ---
STOP_DISTANCE_CM = 6.0
DISTANCE_HYSTERESIS_CM = 1.0 # Derzeit nicht aktiv in der Stopp-Logik

# --- Bewegungskonstanten fuer die Anfangsmanoeuver ---
BASE_POWER_INITIAL_MANEUVER = 20
TURN_ANGLE_SERVO_REVERSE = 45
TURN_TIME_REVERSE = 1.5
TURN_ANGLE_SERVO_FORWARD = -45
TURN_TIME_FORWARD = 1.5

# --- Steuerungs-Variablen und Zustands-Flags ---
last_angle = 0

# Zustaende des Roboters
STATE_RED_LINE_FOLLOWING = 0 # Fahre, folge roten Linien (Default)
STATE_FINISHED = 1           # Auto hat angehalten

current_state = STATE_RED_LINE_FOLLOWING

# Standardgeschwindigkeit fuer das Folgen der roten Linien
SPEED_RED_LINE_DRIVING = 3
# Geschwindigkeit beim Ansteuern des schwarzen Streifens
SPEED_BLACK_STRIPE_DRIVING = 1

# Lenkfaktor fuer das Folgen der roten Linien
STEERING_FACTOR_RED_LINE = 45 # Maximaler Lenkausschlag
# Lenkfaktor fuer das Ansteuern des schwarzen Streifens (geringer fuer Praezision)
STEERING_FACTOR_BLACK_STRIPE = 25 # Reduzierter Lenkfaktor fuer Praezision

# Neuer Lenkfaktor fuer Gegenlenken bei nur einer roten Linie
FIXED_COUNTER_STEER_ANGLE = 15 # Feste Gradzahl fuer das Gegenlenken

# --- Funktionen fuer die Anfangsmanoeuver ---
def stop_car():
    """Stoppt das Auto und setzt die Lenkung auf Mitte."""
    px.stop()
    px.set_dir_servo_angle(0)
    time.sleep(0.5)

def maneuver_car(angle, duration, power, direction='forward'):
    """Bewegt das Auto mit einem bestimmten Lenkwinkel, Dauer und Leistung."""
    print(f"Manoever: Lenkwinkel {angle}, Dauer {duration}s, Leistung {power}, Richtung {direction}")
    px.set_dir_servo_angle(angle)
    if direction == 'forward':
        px.forward(power)
    elif direction == 'backward':
        px.backward(power)
    time.sleep(duration)
    stop_car()

# --- Ultraschallsensor Funktion ---
def get_distance_ultrasonic():
    """
    Liest den Abstand vom Ultraschallsensor in cm.
    Gibt -1 zurueck, wenn ein Lesefehler auftritt oder der Wert ungueltig ist.
    """
    try:
        distance = px.get_distance()
        if distance is not None and distance > 0:
            return distance
        else:
            return -1
    except Exception as e:
        print(f"Fehler beim Auslesen des Ultraschallsensors: {e}")
        return -1

# --- Hauptprogrammablauf ---
def main():
    global last_angle, current_state

    # --- Initialisiere OpenCV Fenster und zeige erste Kamerabilder ---
    initial_frame = picam2.capture_array()
    if initial_frame is None:
        print("Fehler: Kamera konnte kein initiales Bild aufnehmen. Beende.")
        px.stop()
        picam2.stop()
        input("Druecke Enter, um das Terminal zu schliessen...")
        return

    height, width, _ = initial_frame.shape
    dummy_mask = np.zeros((height, width), dtype=np.uint8) # Masken sind 1-Kanal

    # Zeige die Maskenfenster an
    cv2.imshow("Red Mask", dummy_mask)
    cv2.imshow("Black Mask", dummy_mask) # Neues Fenster fuer schwarze Maske
    cv2.waitKey(1)

    print("--- Start der Anfangsmanoeuver ---")
    maneuver_car(TURN_ANGLE_SERVO_REVERSE, TURN_TIME_REVERSE, BASE_POWER_INITIAL_MANEUVER, 'backward')
    maneuver_car(TURN_ANGLE_SERVO_FORWARD, TURN_TIME_FORWARD, BASE_POWER_INITIAL_MANEUVER, 'forward')
    print("--- Anfangsmanoeuver abgeschlossen. Starte Fahrzyklus. ---")

    try:
        while True:
            # Ultraschallsensor-Pruefung fuer den finalen Stopp (immer oberste Prioritaet)
            distance = get_distance_ultrasonic()
            if distance != -1 and distance <= STOP_DISTANCE_CM:
                print(f"Ultraschall: Hindernis bei {distance:.2f} cm erkannt. Stoppe das Auto!")
                current_state = STATE_FINISHED
                px.stop()
                break # Schleife beenden, da Ziel erreicht

            # Wenn das Auto noch nicht gestoppt hat, verarbeite Kamera-Input
            if current_state == STATE_RED_LINE_FOLLOWING:
                frame = picam2.capture_array()
                if frame is None:
                    print("Kamerafehler: Kein Bild erhalten, stoppe.")
                    current_state = STATE_FINISHED
                    break
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                height, width, _ = frame.shape
                frame_center = width // 2

                # --- Rote Linien finden (immer) ---
                roi_y_start = 150
                roi_y_end = 480
                
                roi_hsv_full_width = hsv[roi_y_start:roi_y_end, 0:width]
                red_mask_full_width = cv2.inRange(roi_hsv_full_width, lower_red, upper_red)
                cv2.imshow("Red Mask", red_mask_full_width)

                contours_red, _ = cv2.findContours(red_mask_full_width, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_red = [c for c in contours_red if cv2.contourArea(c) > 100] # Filter kleine Konturen

                # Sortiere rote Konturen, um linke und rechte Linie konsistent zu identifizieren
                contours_red_sorted = []
                if len(contours_red) >= 2:
                    contours_red_sorted = sorted(contours_red, key=lambda c: cv2.boundingRect(c)[0])
                    # Zusaetzliche Ueberpruefung, ob die sortierten Konturen wirklich zwei separate Linien darstellen
                    if len(contours_red_sorted) < 2:
                        # Falls doch weniger als 2 nach Sortierung uebrig bleiben (z.B. weil sie sich ueberlappen)
                        contours_red_sorted = [] # Behandle es als weniger als 2 Linien

                # --- Variablen fuer Steuerung ---
                steering_angle = 0
                drive_power = SPEED_RED_LINE_DRIVING # Standardleistung
                
                black_stripe_found_between_red_lines = False # Flag zur Steuerung der Prioritaet
                center_x_black = None # Initialisiere

                # --- Prioritaet 1: Suche nach dem schwarzen Streifen ZWISCHEN den roten Linien ---
                # Nur suchen, wenn mindestens zwei rote Linien die Spur definieren
                if len(contours_red_sorted) >= 2:
                    left_rect = cv2.boundingRect(contours_red_sorted[0])
                    right_rect = cv2.boundingRect(contours_red_sorted[-1])
                    
                    black_roi_x_start = left_rect[0] + left_rect[2] 
                    black_roi_x_end = right_rect[0]
                    
                    # Stelle sicher, dass der ROI gueltig ist (nicht invertiert oder zu schmal)
                    if black_roi_x_start < black_roi_x_end:
                        roi_hsv_for_black = hsv[roi_y_start:roi_y_end, black_roi_x_start:black_roi_x_end]

                        if roi_hsv_for_black.size > 0:
                            black_mask = cv2.inRange(roi_hsv_for_black, lower_black, upper_black)
                            
                            # Anzeige der Black Mask (nur im definierten ROI)
                            display_black_mask = np.zeros((height, width), dtype=np.uint8)
                            display_black_mask[roi_y_start:roi_y_end, black_roi_x_start:black_roi_x_end] = black_mask
                            cv2.rectangle(display_black_mask, (black_roi_x_start, roi_y_start), 
                                          (black_roi_x_end, roi_y_end), (255), 2)
                            cv2.imshow("Black Mask", display_black_mask)

                            contours_black, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            if contours_black:
                                largest_black_contour = max(contours_black, key=cv2.contourArea)
                                if cv2.contourArea(largest_black_contour) > BLACK_STRIPE_MIN_AREA:
                                    x_black_rel, y_black_rel, w_black, h_black = cv2.boundingRect(largest_black_contour)
                                    center_x_black = (x_black_rel + w_black // 2) + black_roi_x_start
                                    black_stripe_found_between_red_lines = True
                    # else:
                        # print("Debug: Rote Linien zu nah, um einen schwarzen ROI dazwischen zu definieren.")
                
                # Wenn schwarzer Streifen nicht *zwischen* roten Linien gefunden wurde, leere Black Mask anzeigen
                if not black_stripe_found_between_red_lines:
                    cv2.imshow("Black Mask", np.zeros((height, width), dtype=np.uint8)) 


                # --- Steuerung auf Basis der Prioritaeten ---
                if black_stripe_found_between_red_lines:
                    # Schwarzer Streifen hat hoechste Prioritaet, wenn er zwischen den roten Linien ist
                    deviation = center_x_black - frame_center
                    steering_angle = deviation / (width / 2) * STEERING_FACTOR_BLACK_STRIPE 
                    drive_power = SPEED_BLACK_STRIPE_DRIVING
                    print(f"Kamera-Steuerung: Schwarzer Streifen (PRIMAER). Lenkung: {steering_angle:.2f} | Streifen-X: {center_x_black} | US-Dist: {distance:.1f} cm")
                elif len(contours_red_sorted) >= 2:
                    # Zweite Prioritaet: Folgen der roten Linien (wenn Schwarz nicht relevant/sichtbar)
                    left_x = left_rect[0] + left_rect[2] // 2
                    right_x = right_rect[0] + right_rect[2] // 2
                    
                    lane_center = (left_x + right_x) // 2

                    deviation = lane_center - frame_center
                    steering_angle = deviation / (width / 2) * STEERING_FACTOR_RED_LINE 
                    drive_power = SPEED_RED_LINE_DRIVING
                    print(f"Kamera-Steuerung: Rote Linien (PRIMAER - 2 Linien). Lenkung: {steering_angle:.2f} | Mitte-X: {lane_center} | US-Dist: {distance:.1f} cm")

                elif len(contours_red) == 1:
                    # Dritte Prioritaet: Gegenlenken bei nur einer roten Linie
                    x_single, y_single, w_single, h_single = cv2.boundingRect(contours_red[0])
                    center_x_single_red = x_single + w_single // 2

                    deviation = center_x_single_red - frame_center
                    
                    if deviation > 0: # Linie ist rechts von der Mitte, lenke nach links
                        steering_angle = -FIXED_COUNTER_STEER_ANGLE
                    else: # Linie ist links von der Mitte, lenke nach rechts
                        steering_angle = FIXED_COUNTER_STEER_ANGLE
                    
                    drive_power = SPEED_RED_LINE_DRIVING # Behaelt normale Geschwindigkeit
                    print(f"Kamera-Steuerung: Nur 1 rote Linie (GEGENLENKEN). Lenkung: {steering_angle:.2f} | Einzel-X: {center_x_single_red} | US-Dist: {distance:.1f} cm")
                else:
                    # Vierte Prioritaet (Fallback): Keine Linien erkannt, fahre geradeaus
                    steering_angle = 0
                    drive_power = SPEED_RED_LINE_DRIVING
                    print(f"Kamera-Steuerung: KEINE LINIE ERKANNT (FALLBACK - Geradeaus). US-Dist: {distance:.1f} cm")


                # Steuerung anwenden
                smoothed_angle = 0.5 * last_angle + 0.5 * steering_angle
                last_angle = smoothed_angle
                
                px.set_dir_servo_angle(smoothed_angle)
                px.forward(drive_power)


            # Wichtig: cv2.waitKey(1) muss in der Schleife aufgerufen werden,
            # damit die OpenCV-Fenster aktualisiert werden und Ereignisse verarbeitet werden.
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Programm manuell durch 'q' beendet.")
                current_state = STATE_FINISHED
                break

            time.sleep(0.01) # Kleine Pause, um CPU zu entlasten

    except KeyboardInterrupt:
        print("Programm manuell beendet.")
        current_state = STATE_FINISHED

    finally:
        print("Fahrprogramm beendet. Reinige...")
        px.stop()
        px.set_dir_servo_angle(0) # Servo auf Mitte setzen
        picam2.stop()
        print("Hardware gestoppt.")

        # Halte die Fenster offen und das Terminal aktiv
        input("Druecke Enter, um alle Fenster zu schliessen und das Terminal zu beenden...")
        cv2.destroyAllWindows()
        print("Fenster geschlossen. Terminal kann nun beendet werden.")


if __name__ == "__main__":
    main()
