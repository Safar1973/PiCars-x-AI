#!/usr/bin/env python3

from robot_hat.utils import reset_mcu
from picarx import Picarx
from vilib import Vilib
from time import sleep, time, strftime, localtime
import numpy as np
import cv2
import os
import sys

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
MAX_RETRY_COUNT = 3  # Maximale Anzahl von Wiederholungsversuchen

# Status-Variablen
current_state = "START"
start_position = None
current_position = None
obstacle_detected = False
line_detected = False
line_position = 0
retry_count = 0

def setup_camera():
    """Kamera initialisieren und Bildverarbeitung starten"""
    print("Kamera wird initialisiert...")
    try:
        Vilib.camera_start(vflip=False, hflip=False)
        Vilib.display(local=True, web=False)
        
        # Linienerkennung aktivieren - nur color_detect verwenden
        # Rot ist oft die Farbe für Straßenmarkierungen
        Vilib.color_detect("red")
        
        # Objekterkennung aktivieren
        Vilib.object_detect_set_model(model_type="yolo")
        Vilib.object_detect_set_clses(["person", "car", "truck", "bicycle"])
        
        sleep(2)  # Warten auf Kamera-Initialisierung
        print("Kamera bereit!")
        return True
    except Exception as e:
        print(f"Fehler bei der Kamera-Initialisierung: {e}")
        return False

def detect_obstacles():
    """Hindernisse erkennen und Ausweichmanöver einleiten"""
    global obstacle_detected
    
    try:
        # Objekterkennung über Vilib
        # Prüfen, ob Objekterkennung aktiv ist
        if hasattr(Vilib, 'detect_obj_parameter') and Vilib.detect_obj_parameter is not None:
            objects = Vilib.detect_obj_parameter.get('color_n', 0)
            
            if objects > 0:
                # Wenn Objekte erkannt wurden, prüfen ob sie im Weg sind
                obj_x = Vilib.detect_obj_parameter.get('x', 0)
                obj_y = Vilib.detect_obj_parameter.get('y', 0)
                obj_w = Vilib.detect_obj_parameter.get('w', 0)
                
                # Wenn Objekt in der Mitte und groß genug ist, als Hindernis betrachten
                if (CAMERA_WIDTH/3 < obj_x < 2*CAMERA_WIDTH/3) and obj_w > OBSTACLE_THRESHOLD:
                    obstacle_detected = True
                    print("Hindernis erkannt!")
                    return True
        
        obstacle_detected = False
        return False
    except Exception as e:
        print(f"Fehler bei der Hindernisvermeidung: {e}")
        return False

def detect_lane_lines():
    """Fahrspurlinien erkennen und Position bestimmen"""
    global line_detected, line_position
    
    try:
        # Linienerkennung über Vilib color_detect
        # Prüfen, ob Farberkennungsparameter verfügbar sind
        if hasattr(Vilib, 'color_detect_result') and Vilib.color_detect_result is not None:
            # Prüfen, ob Farbobjekte erkannt wurden
            if Vilib.color_detect_result.get('color_n', 0) > 0:
                line_detected = True
                
                # Position der Linie im Bild (x-Koordinate)
                line_x = Vilib.color_detect_result.get('x', CAMERA_WIDTH/2)
                
                # Abweichung von der Bildmitte berechnen (-1 bis 1)
                line_position = (line_x - CAMERA_WIDTH/2) / (CAMERA_WIDTH/2)
                
                return True, line_position
        
        line_detected = False
        return False, 0
    except Exception as e:
        print(f"Fehler bei der Linienerkennung: {e}")
        line_detected = False
        return False, 0

def adjust_steering():
    """Lenkung basierend auf Linienposition anpassen"""
    global line_position
    
    try:
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
    except Exception as e:
        print(f"Fehler bei der Lenkungsanpassung: {e}")
        # Sicherheitshalber geradeaus fahren
        px.set_dir_servo_angle(0)

def emergency_stop():
    """Notfall-Stopp bei kritischen Situationen"""
    print("NOTFALL-STOPP AKTIVIERT!")
    px.stop()
    sleep(1)

def move_forward(distance_cm, speed=30):
    """Vorwärts fahren für eine bestimmte Distanz"""
    global retry_count
    
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
            
            # Ausweichmanöver versuchen
            if retry_count < MAX_RETRY_COUNT:
                retry_count += 1
                print(f"Versuche auszuweichen (Versuch {retry_count}/{MAX_RETRY_COUNT})...")
                
                # Kurz zurücksetzen
                px.backward(speed)
                sleep(1)
                px.stop()
                
                # Leicht zur Seite drehen und erneut versuchen
                turn_angle = 30 if retry_count % 2 == 0 else -30
                turn(turn_angle, 20)
                
                # Erneut vorwärts versuchen
                return move_forward(distance_cm - (elapsed_time * (speed / 3)), speed)
            else:
                print("Maximale Anzahl von Ausweichversuchen erreicht. Breche ab.")
                return False
        
        # Spurhaltung während der Fahrt
        line_found, _ = detect_lane_lines()
        
        # Wenn Linie verloren geht, langsamer werden
        if not line_found and elapsed_time > 1:  # Ignoriere die ersten Sekunden
            print("Warnung: Linie verloren, reduziere Geschwindigkeit")
            px.forward(speed * 0.7)  # Reduzierte Geschwindigkeit
        else:
            px.forward(speed)
        
        # Lenkung anpassen
        adjust_steering()
        
        # Zeit aktualisieren
        elapsed_time = time() - start_time
        sleep(0.1)
    
    # Anhalten nach Erreichen der Distanz
    px.stop()
    print("Vorwärtsbewegung abgeschlossen")
    retry_count = 0  # Reset Wiederholungszähler
    return True

def turn(angle, speed=20):
    """Drehen um einen bestimmten Winkel"""
    print(f"Drehe um {angle}°...")
    
    try:
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
    except Exception as e:
        print(f"Fehler bei der Drehung: {e}")
        emergency_stop()
        return False

def move_backward(distance_cm, speed=20):
    """Rückwärts fahren für eine bestimmte Distanz"""
    print(f"Fahre {distance_cm} cm rückwärts...")
    
    try:
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
            line_found, _ = detect_lane_lines()
            
            # Wenn Linie verloren geht, langsamer werden
            if not line_found and elapsed_time > 1:  # Ignoriere die ersten Sekunden
                print("Warnung: Linie verloren, reduziere Geschwindigkeit")
                px.backward(speed * 0.7)  # Reduzierte Geschwindigkeit
            else:
                px.backward(speed)
            
            # Lenkung anpassen (invertiert für Rückwärtsfahrt)
            if line_detected:
                steering_angle = 30 * line_position  # Invertiert für Rückwärtsfahrt
                steering_angle = max(-30, min(30, steering_angle))
                px.set_dir_servo_angle(steering_angle)
                print(f"Lenkwinkel angepasst (rückwärts): {steering_angle:.1f}°")
            
            # Zeit aktualisieren
            elapsed_time = time() - start_time
            sleep(0.1)
        
        # Anhalten nach Erreichen der Distanz
        px.stop()
        print("Rückwärtsbewegung abgeschlossen")
        return True
    except Exception as e:
        print(f"Fehler bei der Rückwärtsbewegung: {e}")
        emergency_stop()
        return False

def take_photo():
    """Foto aufnehmen und speichern"""
    try:
        _time = strftime('%Y-%m-%d-%H-%M-%S', localtime(time()))
        name = 'autonomous_%s' % _time
        user = os.getlogin()
        user_home = os.path.expanduser(f'~{user}')
        path = f"{user_home}/Pictures/picar-x/"
        
        # Sicherstellen, dass das Verzeichnis existiert
        os.makedirs(path, exist_ok=True)
        
        Vilib.take_photo(name, path)
        print(f'Foto gespeichert als {path}{name}.jpg')
        return True
    except Exception as e:
        print(f"Fehler beim Aufnehmen des Fotos: {e}")
        return False

def check_system():
    """Systemprüfung vor dem Start"""
    print("Führe Systemprüfung durch...")
    
    # Prüfe Kamera
    if not setup_camera():
        print("FEHLER: Kamera konnte nicht initialisiert werden.")
        return False
    
    # Prüfe Motoren durch kurze Bewegung
    try:
        px.forward(10)
        sleep(0.5)
        px.stop()
        sleep(0.5)
        px.backward(10)
        sleep(0.5)
        px.stop()
        print("Motortest erfolgreich.")
    except Exception as e:
        print(f"FEHLER: Motortest fehlgeschlagen: {e}")
        return False
    
    # Prüfe Lenkservo
    try:
        px.set_dir_servo_angle(20)
        sleep(0.5)
        px.set_dir_servo_angle(-20)
        sleep(0.5)
        px.set_dir_servo_angle(0)
        print("Servotest erfolgreich.")
    except Exception as e:
        print(f"FEHLER: Servotest fehlgeschlagen: {e}")
        return False
    
    print("Systemprüfung abgeschlossen. Alle Systeme funktionieren.")
    return True

def main():
    """Hauptfunktion für die autonome Fahrt"""
    try:
        print("=== Autonomes Fahren wird gestartet ===")
        
        # Systemprüfung
        if not check_system():
            print("Systemprüfung fehlgeschlagen. Programm wird beendet.")
            return
        
        # Foto vom Ausgangspunkt machen
        take_photo()
        
        # 1. Vorwärts fahren (2,5 Meter)
        print("\n=== Schritt 1: Vorwärts fahren (2,5 Meter) ===")
        if not move_forward(FORWARD_DISTANCE, SPEED_FORWARD):
            print("Vorwärtsbewegung konnte nicht abgeschlossen werden.")
            return
        sleep(1)
        
        # 2. Drehen (180 Grad)
        print("\n=== Schritt 2: Drehen ===")
        if not turn(TURN_ANGLE, 20):
            print("Drehung konnte nicht abgeschlossen werden.")
            return
        sleep(1)
        
        # 3. Rückwärts fahren (zurück zum Ausgangspunkt)
        print("\n=== Schritt 3: Rückwärts fahren zum Ausgangspunkt ===")
        if not move_backward(FORWARD_DISTANCE, SPEED_BACKWARD):
            print("Rückwärtsbewegung konnte nicht abgeschlossen werden.")
            return
        sleep(1)
        
        # Foto vom Endpunkt machen
        take_photo()
        
        print("\n=== Autonome Fahrt erfolgreich abgeschlossen! ===")
        
    except KeyboardInterrupt:
        print("\nProgramm durch Benutzer unterbrochen.")
    except Exception as e:
        print(f"\nKritischer Fehler: {e}")
        print("Programm wird beendet.")
    finally:
        # Aufräumen
        px.stop()
        try:
            Vilib.camera_close()
        except:
            pass
        print("Programm beendet.")

if __name__ == "__main__":
    main()
