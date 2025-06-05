#!/usr/bin/env python3

from robot_hat.utils import reset_mcu
from picarx import Picarx
from vilib import Vilib
from time import sleep, time
import cv2
import numpy as np

# Reset MCU und Initialisierung
reset_mcu()
sleep(0.2)

# Picarx-Objekt erstellen
px = Picarx()

# Konfigurationswerte
FORWARD_DISTANCE = 250  # 2,5 Meter in cm
SPEED_FORWARD = 20     # Geschwindigkeit vorwärts
SPEED_BACKWARD = 15    # Geschwindigkeit rückwärts
TURN_ANGLE = 180       # Drehwinkel in Grad
CAMERA_WIDTH = 320     # Kamerabreite
CAMERA_HEIGHT = 240    # Kamerahöhe

# Farbe für die Linienerkennung (kann angepasst werden)
LINE_COLOR = "red"     # Mögliche Werte: "red", "green", "blue", "yellow", etc.

# Status-Variablen
line_detected = False
line_position = 0

def setup_camera():
    """Kamera initialisieren"""
    print("Kamera wird initialisiert...")
    try:
        Vilib.camera_start(vflip=False, hflip=False)
        Vilib.display(local=True, web=False)
        
        # Farberkennung aktivieren
        Vilib.color_detect(LINE_COLOR)
        print(f"Farberkennung für {LINE_COLOR} aktiviert.")
        
        sleep(2)  # Warten auf Kamera-Initialisierung
        print("Kamera bereit!")
        return True
    except Exception as e:
        print(f"Fehler bei der Kamera-Initialisierung: {e}")
        return False

def detect_line():
    """Linie mit Vilib-Farberkennung erkennen"""
    global line_detected, line_position
    
    try:
        # Prüfen, ob Farbobjekte erkannt wurden
        if Vilib.detect_obj_parameter['color_n'] > 0:
            line_detected = True
            
            # Position der Linie im Bild (x-Koordinate)
            line_x = Vilib.detect_obj_parameter['color_x']
            
            # Abweichung von der Bildmitte berechnen (-1 bis 1)
            line_position = (line_x - CAMERA_WIDTH/2) / (CAMERA_WIDTH/2)
            
            print(f"Linie erkannt: Position = {line_position:.2f}")
            return True, line_position
        
        line_detected = False
        return False, 0
    except Exception as e:
        print(f"Fehler bei der Linienerkennung: {e}")
        line_detected = False
        return False, 0

def adjust_steering(is_backward=False):
    """Lenkung basierend auf Linienposition anpassen"""
    global line_position
    
    try:
        if line_detected:
            # Lenkwinkel basierend auf Linienposition berechnen
            multiplier = 30  # Empfindlichkeit der Lenkung
            
            if is_backward:
                # Invertierte Lenkung für Rückwärtsfahrt
                steering_angle = multiplier * line_position
            else:
                # Normale Lenkung für Vorwärtsfahrt
                steering_angle = -multiplier * line_position
            
            # Lenkwinkel begrenzen
            steering_angle = max(-30, min(30, steering_angle))
            
            # Lenkwinkel setzen
            px.set_dir_servo_angle(steering_angle)
            
            direction = "rückwärts" if is_backward else "vorwärts"
            print(f"Lenkwinkel angepasst ({direction}): {steering_angle:.1f}°")
            return True
        else:
            # Wenn keine Linie erkannt, geradeaus fahren
            px.set_dir_servo_angle(0)
            print("Keine Linie erkannt, fahre geradeaus")
            return False
    except Exception as e:
        print(f"Fehler bei der Lenkungsanpassung: {e}")
        # Sicherheitshalber geradeaus fahren
        px.set_dir_servo_angle(0)
        return False

def drive_forward(distance_cm, speed=20):
    """Der Linie folgen für eine bestimmte Distanz"""
    print(f"Fahre {distance_cm} cm vorwärts...")
    
    # Startzeit merken
    start_time = time()
    
    # Geschätzte Zeit für die Distanz berechnen
    # Annahme: 10 cm/s bei Geschwindigkeit 20
    estimated_time = distance_cm / (speed / 5)
    
    # Vorwärts fahren starten
    px.forward(speed)
    
    # Fahren bis Distanz erreicht
    elapsed_time = 0
    line_lost_count = 0
    max_line_lost = 20  # Maximale Anzahl von Frames, in denen die Linie verloren sein darf
    
    while elapsed_time < estimated_time:
        # Linie erkennen
        line_found, _ = detect_line()
        
        if line_found:
            # Lenkung anpassen
            adjust_steering(is_backward=False)
            line_lost_count = 0  # Zurücksetzen des Zählers
        else:
            line_lost_count += 1
            print(f"Linie verloren ({line_lost_count}/{max_line_lost})")
            
            # Wenn die Linie zu lange verloren ist, langsamer werden
            if line_lost_count > 5:
                px.forward(speed * 0.7)  # Reduzierte Geschwindigkeit
            
            # Wenn die Linie zu lange verloren ist, abbrechen
            if line_lost_count > max_line_lost:
                print("Linie dauerhaft verloren, stoppe...")
                px.stop()
                return False
        
        # Zeit aktualisieren
        elapsed_time = time() - start_time
        sleep(0.05)  # Kürzere Wartezeit für schnellere Reaktion
    
    # Anhalten nach Erreichen der Distanz
    px.stop()
    print("Vorwärtsbewegung abgeschlossen")
    return True

def turn(angle, speed=15):
    """Drehen um einen bestimmten Winkel"""
    print(f"Drehe um {angle}°...")
    
    try:
        # Richtung bestimmen
        direction = 1 if angle > 0 else -1
        angle = abs(angle)
        
        # Geschätzte Zeit für die Drehung berechnen
        # Annahme: 90° bei Geschwindigkeit 15 dauert ca. 2.5 Sekunden
        estimated_time = (angle / 90) * 2.5
        
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
        
        # Kurze Pause, um die Kamera zu stabilisieren
        sleep(0.5)
        
        print("Drehung abgeschlossen")
        return True
    except Exception as e:
        print(f"Fehler bei der Drehung: {e}")
        px.stop()
        return False

def drive_backward(distance_cm, speed=15):
    """Der Linie rückwärts folgen für eine bestimmte Distanz"""
    print(f"Fahre {distance_cm} cm rückwärts...")
    
    # Startzeit merken
    start_time = time()
    
    # Geschätzte Zeit für die Distanz berechnen
    # Annahme: 8 cm/s bei Geschwindigkeit 15
    estimated_time = distance_cm / (speed / 5)
    
    # Rückwärts fahren starten
    px.backward(speed)
    
    # Fahren bis Distanz erreicht
    elapsed_time = 0
    line_lost_count = 0
    max_line_lost = 20  # Maximale Anzahl von Frames, in denen die Linie verloren sein darf
    
    while elapsed_time < estimated_time:
        # Linie erkennen
        line_found, _ = detect_line()
        
        if line_found:
            # Lenkung anpassen (invertiert für Rückwärtsfahrt)
            adjust_steering(is_backward=True)
            line_lost_count = 0  # Zurücksetzen des Zählers
        else:
            line_lost_count += 1
            print(f"Linie verloren ({line_lost_count}/{max_line_lost})")
            
            # Wenn die Linie zu lange verloren ist, langsamer werden
            if line_lost_count > 5:
                px.backward(speed * 0.7)  # Reduzierte Geschwindigkeit
            
            # Wenn die Linie zu lange verloren ist, abbrechen
            if line_lost_count > max_line_lost:
                print("Linie dauerhaft verloren, stoppe...")
                px.stop()
                return False
        
        # Zeit aktualisieren
        elapsed_time = time() - start_time
        sleep(0.05)  # Kürzere Wartezeit für schnellere Reaktion
    
    # Anhalten nach Erreichen der Distanz
    px.stop()
    print("Rückwärtsbewegung abgeschlossen")
    return True

def main():
    """Hauptfunktion für die autonome Fahrt"""
    try:
        print("=== Autonomes Linienfolgen wird gestartet ===")
        
        # Kamera initialisieren
        if not setup_camera():
            print("Kamera konnte nicht initialisiert werden. Programm wird beendet.")
            return
        
        # Kurze Pause, um die Kamera zu stabilisieren
        sleep(2)
        
        # 1. Vorwärts der Linie folgen (2,5 Meter)
        print("\n=== Schritt 1: Der Linie 2,5 Meter vorwärts folgen ===")
        if not drive_forward(FORWARD_DISTANCE, SPEED_FORWARD):
            print("Linienfolgen vorwärts konnte nicht abgeschlossen werden.")
            return
        sleep(1)
        
        # 2. Drehen (180 Grad)
        print("\n=== Schritt 2: Drehen ===")
        if not turn(TURN_ANGLE, 15):
            print("Drehung konnte nicht abgeschlossen werden.")
            return
        sleep(1)
        
        # 3. Rückwärts der Linie folgen (zurück zum Ausgangspunkt)
        print("\n=== Schritt 3: Der Linie 2,5 Meter rückwärts folgen ===")
        if not drive_backward(FORWARD_DISTANCE, SPEED_BACKWARD):
            print("Linienfolgen rückwärts konnte nicht abgeschlossen werden.")
            return
        sleep(1)
        
        print("\n=== Autonomes Linienfolgen erfolgreich abgeschlossen! ===")
        
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
