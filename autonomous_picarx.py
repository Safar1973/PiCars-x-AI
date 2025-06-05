#!/usr/bin/env python3

from robot_hat.utils import reset_mcu
from picarx import Picarx
from vilib import Vilib
from time import sleep, time, strftime, localtime
import numpy as np
import cv2
import os

# Reset MCU und Initialisierung
reset_mcu()
sleep(0.2)

# Picarx-Objekt erstellen
px = Picarx()

# Konfigurationswerte
FORWARD_DISTANCE = 250  # 2,5 Meter in cm
SPEED_FORWARD = 30
SPEED_BACKWARD = 20
TURN_ANGLE = 180  # Drehwinkel in Grad
LINE_THRESHOLD = 60  # Schwellenwert für Linienerkennung
OBSTACLE_THRESHOLD = 30  # Schwellenwert für Hindernisvermeidung in cm
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

# Status-Variablen
current_state = "START"
start_position = None
current_position = None
obstacle_detected = False
line_detected = False
line_position = 0

def setup_camera():
    """Kamera initialisieren und Bildverarbeitung starten"""
    print("Kamera wird initialisiert...")
    Vilib.camera_start(vflip=False, hflip=False)
    Vilib.display(local=True, web=False)
    
    # Linienerkennung aktivieren
    Vilib.color_detect("red")  # Für rote Linien, kann je nach Straßenmarkierung angepasst werden
    Vilib.detect_color_name("red")
    
    # Objekterkennung aktivieren
    Vilib.object_detect_set_model(model_type="yolo")
    Vilib.object_detect_set_clses(["person", "car", "truck", "bicycle"])
    
    sleep(2)  # Warten auf Kamera-Initialisierung
    print("Kamera bereit!")

def detect_obstacles():
    """Hindernisse erkennen und Ausweichmanöver einleiten"""
    global obstacle_detected
    
    # Objekterkennung über Vilib
    objects = Vilib.detect_obj_parameter['color_n']
    
    if objects > 0:
        # Wenn Objekte erkannt wurden, prüfen ob sie im Weg sind
        obj_x = Vilib.detect_obj_parameter['x']
        obj_y = Vilib.detect_obj_parameter['y']
        obj_w = Vilib.detect_obj_parameter['w']
        
        # Wenn Objekt in der Mitte und groß genug ist, als Hindernis betrachten
        if (CAMERA_WIDTH/3 < obj_x < 2*CAMERA_WIDTH/3) and obj_w > OBSTACLE_THRESHOLD:
            obstacle_detected = True
            print("Hindernis erkannt!")
            return True
    
    obstacle_detected = False
    return False

def detect_lane_lines():
    """Fahrspurlinien erkennen und Position bestimmen"""
    global line_detected, line_position
    
    # Linienerkennung über Vilib
    if Vilib.detect_obj_parameter['color_n'] > 0:
        line_detected = True
        
        # Position der Linie im Bild (x-Koordinate)
        line_x = Vilib.detect_obj_parameter['x']
        
        # Abweichung von der Bildmitte berechnen (-1 bis 1)
        line_position = (line_x - CAMERA_WIDTH/2) / (CAMERA_WIDTH/2)
        
        return True, line_position
    
    line_detected = False
    return False, 0

def adjust_steering():
    """Lenkung basierend auf Linienposition anpassen"""
    global line_position
    
    if line_detected:
        # Lenkwinkel basierend auf Linienposition berechnen
        # Multiplikator kann angepasst werden, um Empfindlichkeit zu steuern
        steering_angle = -30 * line_position
        
        # Lenkwinkel begrenzen
        steering_angle = max(-30, min(30, steering_angle))
        
        # Lenkwinkel setzen
        px.set_dir_servo_angle(steering_angle)
        print(f"Lenkwinkel angepasst: {steering_angle:.1f}°")
    else:
        # Wenn keine Linie erkannt, geradeaus fahren
        px.set_dir_servo_angle(0)

def move_forward(distance_cm, speed=30):
    """Vorwärts fahren für eine bestimmte Distanz"""
    print(f"Fahre {distance_cm} cm vorwärts...")
    
    # Startzeit merken
    start_time = time()
    
    # Geschätzte Zeit für die Distanz berechnen (kann je nach Roboter angepasst werden)
    # Annahme: 10 cm/s bei Geschwindigkeit 30
    estimated_time = distance_cm / (speed / 3)
    
    # Vorwärts fahren starten
    px.set_dir_servo_angle(0)
    px.forward(speed)
    
    # Fahren bis Distanz erreicht oder Hindernis erkannt
    elapsed_time = 0
    while elapsed_time < estimated_time:
        # Prüfen auf Hindernisse
        if detect_obstacles():
            print("Hindernis erkannt, stoppe...")
            px.stop()
            return False
        
        # Spurhaltung während der Fahrt
        detect_lane_lines()
        adjust_steering()
        
        # Zeit aktualisieren
        elapsed_time = time() - start_time
        sleep(0.1)
    
    # Anhalten nach Erreichen der Distanz
    px.stop()
    print("Vorwärtsbewegung abgeschlossen")
    return True

def turn(angle, speed=20):
    """Drehen um einen bestimmten Winkel"""
    print(f"Drehe um {angle}°...")
    
    # Richtung bestimmen
    direction = 1 if angle > 0 else -1
    angle = abs(angle)
    
    # Geschätzte Zeit für die Drehung berechnen (kann je nach Roboter angepasst werden)
    # Annahme: 90° bei Geschwindigkeit 20 dauert ca. 2 Sekunden
    estimated_time = (angle / 90) * 2
    
    # Drehung starten
    if direction > 0:
        px.set_dir_servo_angle(30)  # Maximaler Lenkwinkel
        px.forward(speed)
    else:
        px.set_dir_servo_angle(-30)  # Maximaler Lenkwinkel in die andere Richtung
        px.forward(speed)
    
    # Warten bis Drehung abgeschlossen
    sleep(estimated_time)
    
    # Anhalten nach Drehung
    px.stop()
    px.set_dir_servo_angle(0)
    print("Drehung abgeschlossen")
    return True

def move_backward(distance_cm, speed=20):
    """Rückwärts fahren für eine bestimmte Distanz"""
    print(f"Fahre {distance_cm} cm rückwärts...")
    
    # Startzeit merken
    start_time = time()
    
    # Geschätzte Zeit für die Distanz berechnen
    # Annahme: 8 cm/s bei Geschwindigkeit 20
    estimated_time = distance_cm / (speed / 2.5)
    
    # Rückwärts fahren starten
    px.set_dir_servo_angle(0)
    px.backward(speed)
    
    # Fahren bis Distanz erreicht
    elapsed_time = 0
    while elapsed_time < estimated_time:
        # Spurhaltung während der Fahrt
        detect_lane_lines()
        adjust_steering()
        
        # Zeit aktualisieren
        elapsed_time = time() - start_time
        sleep(0.1)
    
    # Anhalten nach Erreichen der Distanz
    px.stop()
    print("Rückwärtsbewegung abgeschlossen")
    return True

def take_photo():
    """Foto aufnehmen und speichern"""
    _time = strftime('%Y-%m-%d-%H-%M-%S', localtime(time()))
    name = 'autonomous_%s' % _time
    user = os.getlogin()
    user_home = os.path.expanduser(f'~{user}')
    path = f"{user_home}/Pictures/picar-x/"
    
    # Sicherstellen, dass das Verzeichnis existiert
    os.makedirs(path, exist_ok=True)
    
    Vilib.take_photo(name, path)
    print(f'Foto gespeichert als {path}{name}.jpg')

def main():
    """Hauptfunktion für die autonome Fahrt"""
    try:
        print("Autonomes Fahren wird gestartet...")
        
        # Kamera initialisieren
        setup_camera()
        sleep(1)
        
        # Foto vom Ausgangspunkt machen
        take_photo()
        
        # 1. Vorwärts fahren (2,5 Meter)
        print("Schritt 1: Vorwärts fahren (2,5 Meter)")
        move_forward(FORWARD_DISTANCE, SPEED_FORWARD)
        sleep(1)
        
        # 2. Drehen (180 Grad)
        print("Schritt 2: Drehen")
        turn(TURN_ANGLE, 20)
        sleep(1)
        
        # 3. Rückwärts fahren (zurück zum Ausgangspunkt)
        print("Schritt 3: Rückwärts fahren zum Ausgangspunkt")
        move_backward(FORWARD_DISTANCE, SPEED_BACKWARD)
        sleep(1)
        
        # Foto vom Endpunkt machen
        take_photo()
        
        print("Autonome Fahrt abgeschlossen!")
        
    except Exception as e:
        print(f"Fehler: {e}")
    finally:
        # Aufräumen
        px.stop()
        Vilib.camera_close()
        print("Programm beendet.")

if __name__ == "__main__":
    main()
