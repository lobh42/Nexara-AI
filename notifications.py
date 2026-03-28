"""
Notification Module
Send alerts to technicians via email, telegram, or in-app.
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any


def get_default_settings() -> Dict:
    """Get default notification settings."""
    return {
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "smtp_from_email": "",
        "smtp_from_name": "Nexara AI",
        "smtp_username": "",
        "smtp_password": "",
    }


def send_email_notification(
    to_email: str,
    subject: str,
    body: str,
    html_body: str = None,
    settings: Dict = None
) -> Dict[str, Any]:
    """Send email notification via SMTP."""
    if settings is None:
        settings = get_default_settings()
    
    smtp_host = settings.get("smtp_host", "")
    smtp_port = int(settings.get("smtp_port", 587))
    smtp_username = settings.get("smtp_username", "")
    smtp_password = settings.get("smtp_password", "")
    from_email = settings.get("smtp_from_email", "")
    from_name = settings.get("smtp_from_name", "Nexara AI")
    
    if not smtp_host or not smtp_username or not from_email:
        return {"success": False, "message": "SMTP not configured"}
    
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{from_name} <{from_email}>"
        msg["To"] = to_email
        
        text_part = MIMEText(body, "plain")
        msg.attach(text_part)
        
        if html_body:
            html_part = MIMEText(html_body, "html")
            msg.attach(html_part)
        
        context = ssl.create_default_context()
        
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls(context=context)
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        
        return {"success": True, "message": "Email sent"}
    
    except Exception as e:
        return {"success": False, "message": str(e)}


def build_alert_html(equipment_name: str, failure_prob: float, risk_level: str, message: str) -> str:
    """Build HTML email for alerts."""
    color = {"CRITICAL": "#DC2626", "HIGH": "#D97706", "MEDIUM": "#2563EB"}.get(risk_level, "#64748B")
    
    return f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <div style="background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%); padding: 20px; border-radius: 10px 10px 0 0;">
            <h2 style="color: white; margin: 0;">⚠️ Nexara AI Maintenance Alert</h2>
        </div>
        <div style="background: #F8FAFC; padding: 20px; border: 1px solid #E2E8F0;">
            <p style="font-size: 16px;">Hi,</p>
            <p style="font-size: 16px;">Equipment <strong>{equipment_name}</strong> requires immediate attention.</p>
            
            <div style="background: white; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <table style="width: 100%;">
                    <tr>
                        <td style="padding: 8px 0;"><strong>Failure Probability:</strong></td>
                        <td style="padding: 8px 0; color: {color}; font-weight: bold;">{failure_prob}%</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0;"><strong>Risk Level:</strong></td>
                        <td style="padding: 8px 0; color: {color}; font-weight: bold;">{risk_level}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0;"><strong>Message:</strong></td>
                        <td style="padding: 8px 0;">{message}</td>
                    </tr>
                </table>
            </div>
            
            <p style="font-size: 14px; color: #64748B;">Please take immediate action to prevent equipment failure.</p>
        </div>
        <div style="background: #1E293B; padding: 15px; border-radius: 0 0 10px 10px; text-align: center;">
            <p style="color: #94A3B8; font-size: 12px; margin: 0;">Sent by Nexara AI - Predictive Maintenance System</p>
        </div>
    </body>
    </html>
    """


def build_alert_text(equipment_name: str, failure_prob: float, risk_level: str, message: str) -> str:
    """Build plain text email for alerts."""
    return f"""
Nexara AI Maintenance Alert

Equipment: {equipment_name}
Failure Probability: {failure_prob}%
Risk Level: {risk_level}

Message: {message}

Please take immediate action to prevent equipment failure.

Sent by Nexara AI - Predictive Maintenance System
"""


def notify_technicians(
    equipment_id: str,
    equipment_name: str,
    failure_probability: float,
    risk_level: str,
    message: str = "",
    notify_types: List[str] = None,
    technicians: List[Dict] = None,
    settings: Dict = None
) -> Dict[str, Any]:
    """Send notifications to technicians."""
    if notify_types is None:
        notify_types = ["email"]
    
    if technicians is None:
        # Fallback to a default if no technicians are provided
        return {"success": False, "message": "No technicians configured"}
    
    results = {"email": {"sent": 0, "failed": 0}}
    
    subject = f"⚠️ Nexara AI ALERT: {equipment_name} - {risk_level} Risk"
    html_body = build_alert_html(equipment_name, failure_probability, risk_level, message)
    text_body = build_alert_text(equipment_name, failure_probability, risk_level, message)
    
    for tech in technicians:
        if not tech.get("alert_enabled", True):
            continue
        
        email = tech.get("email")
        if "email" in notify_types and email:
            result = send_email_notification(
                email,
                subject,
                text_body,
                html_body,
                settings
            )
            
            if result["success"]:
                results["email"]["sent"] += 1
            else:
                results["email"]["failed"] += 1
    
    return {
        "success": True,
        "results": results,
        "message": f"Email: {results['email']['sent']} sent, {results['email']['failed']} failed"
    }


def check_and_notify(failure_predictions: List[Dict], threshold: float = 70.0, technician_config: Dict = None) -> Dict:
    """Check failure predictions and notify if above threshold."""
    notifications_sent = []
    
    if technician_config and technician_config.get("email"):
        # Create a technician list from the config
        technicians = [technician_config]
    else:
        return {"success": False, "notifications_sent": 0}
    
    for pred in failure_predictions:
        prob = pred.get("failure_probability", 0)
        
        if prob >= threshold:
            risk = pred.get("risk_level", "MEDIUM")
            equipment_id = pred.get("equipment_id", "")
            equipment_name = pred.get("equipment_name", "")
            action = pred.get("recommended_action", "Please check equipment")
            
            result = notify_technicians(
                equipment_id,
                equipment_name,
                prob,
                risk,
                action,
                technicians=technicians
            )
            
            notifications_sent.append({
                "equipment": equipment_name,
                "probability": prob,
                "risk": risk,
                "result": result
            })
    
    return {
        "success": True,
        "notifications_sent": len(notifications_sent),
        "details": notifications_sent
    }


def test_email_notification(email: str) -> Dict:
    """Send a test email."""
    return send_email_notification(
        email,
        "Test Email - Nexara AI",
        "This is a test email from Nexara AI Predictive Maintenance System."
    )
