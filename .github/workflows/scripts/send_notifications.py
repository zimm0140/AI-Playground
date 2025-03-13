#!/usr/bin/env python
"""
CI Notification Sender

This script sends notifications about CI results to different channels:
- Email (via SMTP)
- Slack (via webhook)
- Microsoft Teams (via webhook)

It formats the notification based on the CI results and provides links
to view detailed reports.
"""

import argparse
import json
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests


class NotificationSender:
    def __init__(self, metrics_file=None, artifacts_dir="ci_artifacts"):
        self.artifacts_dir = artifacts_dir
        self.metrics_file = metrics_file or os.path.join(
            artifacts_dir, "metrics/ci_metrics.json"
        )

        # GitHub Actions environment variables
        self.github_server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
        self.github_repo = os.environ.get("GITHUB_REPOSITORY", "")
        self.github_run_id = os.environ.get("GITHUB_RUN_ID", "")
        self.github_workflow = os.environ.get("GITHUB_WORKFLOW", "CI")
        self.github_actor = os.environ.get("GITHUB_ACTOR", "")
        self.github_ref = os.environ.get("GITHUB_REF", "")
        self.github_sha = os.environ.get("GITHUB_SHA", "")
        self.github_run_number = os.environ.get("GITHUB_RUN_NUMBER", "")

        # Construct useful URLs
        self.run_url = (
            f"{self.github_server}/{self.github_repo}/actions/runs/{self.github_run_id}"
        )
        self.commit_url = (
            f"{self.github_server}/{self.github_repo}/commit/{self.github_sha}"
        )

        # Notification settings
        self.notify_on_success = False  # Only notify on failure by default
        self.notification_title = ""
        self.notification_body = ""
        self.notification_summary = ""
        self.status = "unknown"

        # Load metrics if available
        self.metrics = self._load_metrics()

    def _load_metrics(self):
        """Load metrics from the metrics file."""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file) as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading metrics file: {str(e)}")
        return {}

    def prepare_notification_content(self, status="success"):
        """Prepare the notification content based on CI status."""
        self.status = status

        # Determine branch or PR name
        ref_name = self.github_ref
        if ref_name.startswith("refs/heads/"):
            ref_name = ref_name.replace("refs/heads/", "")
        elif ref_name.startswith("refs/pull/"):
            ref_name = f"PR #{ref_name.split('/')[2]}"

        # Prepare notification title
        if status == "success":
            self.notification_title = (
                f"✅ CI Succeeded: {self.github_workflow} #{self.github_run_number}"
            )
        else:
            self.notification_title = (
                f"❌ CI Failed: {self.github_workflow} #{self.github_run_number}"
            )

        # Extract metrics for the notification
        test_metrics = self.metrics.get("test_metrics", {})
        test_pass_rate = test_metrics.get("pass_rate", 0)
        test_failures = test_metrics.get("failed_tests", 0)

        coverage_metrics = self.metrics.get("coverage_metrics", {})
        coverage_rate = coverage_metrics.get("overall_coverage", 0)

        security_metrics = self.metrics.get("security_metrics", {})
        vulnerabilities = security_metrics.get("total_vulnerabilities", 0)

        platform_metrics = self.metrics.get("platform_metrics", {})
        platform_issues = platform_metrics.get("total_issues", 0)

        # Prepare notification body (HTML format)
        self.notification_body = f"""
        <h2>{self.notification_title}</h2>
        <p>
            <b>Workflow:</b> {self.github_workflow}<br>
            <b>Run:</b> <a href="{self.run_url}">#{self.github_run_number}</a><br>
            <b>Branch/Reference:</b> {ref_name}<br>
            <b>Triggered by:</b> {self.github_actor}<br>
            <b>Commit:</b> <a href="{self.commit_url}">{self.github_sha[:7]}</a><br>
            <b>Status:</b> {status.upper()}<br>
            <b>Completed at:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </p>

        <h3>Summary</h3>
        <ul>
            <li>Test Pass Rate: {test_pass_rate}%</li>
            <li>Test Failures: {test_failures}</li>
            <li>Code Coverage: {coverage_rate}%</li>
            <li>Security Vulnerabilities: {vulnerabilities}</li>
            <li>Platform Compatibility Issues: {platform_issues}</li>
        </ul>

        <p>
            <a href="{self.run_url}">View Details</a>
        </p>
        """

        # Prepare notification summary (plain text)
        self.notification_summary = f"""
{self.notification_title}

Workflow: {self.github_workflow}
Run: #{self.github_run_number}
Branch/Reference: {ref_name}
Triggered by: {self.github_actor}
Commit: {self.github_sha[:7]}
Status: {status.upper()}
Completed at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Summary:
- Test Pass Rate: {test_pass_rate}%
- Test Failures: {test_failures}
- Code Coverage: {coverage_rate}%
- Security Vulnerabilities: {vulnerabilities}
- Platform Compatibility Issues: {platform_issues}

View Details: {self.run_url}
        """

    def send_email(
        self,
        recipients,
        smtp_server,
        smtp_port=587,
        smtp_user=None,
        smtp_password=None,
        sender=None,
    ):
        """Send a notification via email using SMTP."""
        if not recipients:
            print("No email recipients specified")
            return False

        if not sender:
            sender = f"CI Notifications <noreply@{self.github_repo.split('/')[0]}>"

        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = self.notification_title
            msg["From"] = sender

            if isinstance(recipients, list):
                msg["To"] = ", ".join(recipients)
            else:
                msg["To"] = recipients

            # Attach plain text and HTML versions
            text_part = MIMEText(self.notification_summary, "plain")
            html_part = MIMEText(self.notification_body, "html")
            msg.attach(text_part)
            msg.attach(html_part)

            # Connect to SMTP server
            if smtp_port == 465:
                server = smtplib.SMTP_SSL(smtp_server, smtp_port)
            else:
                server = smtplib.SMTP(smtp_server, smtp_port)
                server.ehlo()
                server.starttls()
                server.ehlo()

            # Login if credentials provided
            if smtp_user and smtp_password:
                server.login(smtp_user, smtp_password)

            # Send email
            server.sendmail(sender, recipients, msg.as_string())
            server.quit()

            print(f"Email notification sent to {recipients}")
            return True

        except Exception as e:
            print(f"Error sending email notification: {str(e)}")
            return False

    def send_slack(self, webhook_url):
        """Send a notification to Slack using a webhook."""
        if not webhook_url:
            print("No Slack webhook URL specified")
            return False

        try:
            # Format the message for Slack
            if self.status == "success":
                color = "good"  # green
            else:
                color = "danger"  # red

            # Extract metrics for the notification
            test_metrics = self.metrics.get("test_metrics", {})
            test_pass_rate = test_metrics.get("pass_rate", 0)
            test_failures = test_metrics.get("failed_tests", 0)

            coverage_metrics = self.metrics.get("coverage_metrics", {})
            coverage_rate = coverage_metrics.get("overall_coverage", 0)

            security_metrics = self.metrics.get("security_metrics", {})
            vulnerabilities = security_metrics.get("total_vulnerabilities", 0)

            # Construct the Slack message
            message = {
                "attachments": [
                    {
                        "color": color,
                        "title": self.notification_title,
                        "title_link": self.run_url,
                        "fields": [
                            {
                                "title": "Workflow",
                                "value": self.github_workflow,
                                "short": True,
                            },
                            {
                                "title": "Run",
                                "value": f"<{self.run_url}|#{self.github_run_number}>",
                                "short": True,
                            },
                            {
                                "title": "Branch/Reference",
                                "value": self.github_ref.replace("refs/heads/", ""),
                                "short": True,
                            },
                            {
                                "title": "Triggered by",
                                "value": self.github_actor,
                                "short": True,
                            },
                            {
                                "title": "Commit",
                                "value": f"<{self.commit_url}|{self.github_sha[:7]}>",
                                "short": True,
                            },
                            {
                                "title": "Status",
                                "value": self.status.upper(),
                                "short": True,
                            },
                            {
                                "title": "Test Pass Rate",
                                "value": f"{test_pass_rate}%",
                                "short": True,
                            },
                            {
                                "title": "Test Failures",
                                "value": str(test_failures),
                                "short": True,
                            },
                            {
                                "title": "Code Coverage",
                                "value": f"{coverage_rate}%",
                                "short": True,
                            },
                            {
                                "title": "Security Vulnerabilities",
                                "value": str(vulnerabilities),
                                "short": True,
                            },
                        ],
                        "footer": f"CI Notification | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    }
                ]
            }

            # Send the message to Slack
            response = requests.post(webhook_url, json=message)

            if response.status_code == 200:
                print("Slack notification sent successfully")
                return True
            else:
                print(
                    f"Error sending Slack notification: {response.status_code} {response.text}"
                )
                return False

        except Exception as e:
            print(f"Error sending Slack notification: {str(e)}")
            return False

    def send_teams(self, webhook_url):
        """Send a notification to Microsoft Teams using a webhook."""
        if not webhook_url:
            print("No Teams webhook URL specified")
            return False

        try:
            # Format the message for Teams
            if self.status == "success":
                theme_color = "00FF00"  # green
            else:
                theme_color = "FF0000"  # red

            # Extract metrics for the notification
            test_metrics = self.metrics.get("test_metrics", {})
            test_pass_rate = test_metrics.get("pass_rate", 0)
            test_failures = test_metrics.get("failed_tests", 0)

            coverage_metrics = self.metrics.get("coverage_metrics", {})
            coverage_rate = coverage_metrics.get("overall_coverage", 0)

            security_metrics = self.metrics.get("security_metrics", {})
            vulnerabilities = security_metrics.get("total_vulnerabilities", 0)

            # Construct the Teams message
            message = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": theme_color,
                "summary": self.notification_title,
                "sections": [
                    {
                        "activityTitle": self.notification_title,
                        "facts": [
                            {"name": "Workflow", "value": self.github_workflow},
                            {"name": "Run", "value": f"#{self.github_run_number}"},
                            {
                                "name": "Branch/Reference",
                                "value": self.github_ref.replace("refs/heads/", ""),
                            },
                            {"name": "Triggered by", "value": self.github_actor},
                            {"name": "Commit", "value": self.github_sha[:7]},
                            {"name": "Status", "value": self.status.upper()},
                            {"name": "Test Pass Rate", "value": f"{test_pass_rate}%"},
                            {"name": "Test Failures", "value": str(test_failures)},
                            {"name": "Code Coverage", "value": f"{coverage_rate}%"},
                            {
                                "name": "Security Vulnerabilities",
                                "value": str(vulnerabilities),
                            },
                        ],
                        "markdown": True,
                    }
                ],
                "potentialAction": [
                    {
                        "@type": "OpenUri",
                        "name": "View Run",
                        "targets": [{"os": "default", "uri": self.run_url}],
                    },
                    {
                        "@type": "OpenUri",
                        "name": "View Commit",
                        "targets": [{"os": "default", "uri": self.commit_url}],
                    },
                ],
            }

            # Send the message to Teams
            response = requests.post(webhook_url, json=message)

            if response.status_code < 400:  # Teams returns 200 or 201 for success
                print("Teams notification sent successfully")
                return True
            else:
                print(
                    f"Error sending Teams notification: {response.status_code} {response.text}"
                )
                return False

        except Exception as e:
            print(f"Error sending Teams notification: {str(e)}")
            return False

    def run(self, status="success", notify_channels=None):
        """Run the notification sender."""
        self.prepare_notification_content(status)

        # Only notify on failure unless explicitly configured to notify on success
        if status != "success" or self.notify_on_success:
            # Check for notification channels
            if not notify_channels:
                print("No notification channels specified")
                return

            # Send notifications to all specified channels
            for channel, config in notify_channels.items():
                if channel == "email" and config.get("enabled", False):
                    self.send_email(
                        recipients=config.get("recipients"),
                        smtp_server=config.get("smtp_server"),
                        smtp_port=config.get("smtp_port", 587),
                        smtp_user=config.get("smtp_user"),
                        smtp_password=config.get("smtp_password"),
                        sender=config.get("sender"),
                    )

                elif channel == "slack" and config.get("enabled", False):
                    self.send_slack(webhook_url=config.get("webhook_url"))

                elif channel == "teams" and config.get("enabled", False):
                    self.send_teams(webhook_url=config.get("webhook_url"))

        else:
            print("Skipping notifications for successful run")


def main():
    parser = argparse.ArgumentParser(description="Send CI notifications")
    parser.add_argument("--metrics-file", help="Path to metrics JSON file")
    parser.add_argument(
        "--artifacts-dir",
        default="ci_artifacts",
        help="Directory containing CI artifacts",
    )
    parser.add_argument(
        "--status",
        default="success",
        choices=["success", "failure"],
        help="CI run status",
    )
    parser.add_argument(
        "--notify-success",
        action="store_true",
        help="Send notifications on success (default: only on failure)",
    )
    parser.add_argument("--email", action="store_true", help="Send email notification")
    parser.add_argument("--email-recipients", help="Email recipients (comma-separated)")
    parser.add_argument("--email-server", help="SMTP server")
    parser.add_argument("--email-port", type=int, default=587, help="SMTP port")
    parser.add_argument("--email-user", help="SMTP username")
    parser.add_argument("--email-password", help="SMTP password")
    parser.add_argument("--slack", action="store_true", help="Send Slack notification")
    parser.add_argument("--slack-webhook", help="Slack webhook URL")
    parser.add_argument("--teams", action="store_true", help="Send Teams notification")
    parser.add_argument("--teams-webhook", help="Teams webhook URL")
    args = parser.parse_args()

    # Initialize notification sender
    sender = NotificationSender(
        metrics_file=args.metrics_file, artifacts_dir=args.artifacts_dir
    )

    # Configure notification channels
    notify_channels = {}

    if args.email:
        notify_channels["email"] = {
            "enabled": True,
            "recipients": args.email_recipients.split(",")
            if args.email_recipients
            else [],
            "smtp_server": args.email_server,
            "smtp_port": args.email_port,
            "smtp_user": args.email_user,
            "smtp_password": args.email_password,
        }

    if args.slack:
        notify_channels["slack"] = {"enabled": True, "webhook_url": args.slack_webhook}

    if args.teams:
        notify_channels["teams"] = {"enabled": True, "webhook_url": args.teams_webhook}

    # Set notification preferences
    sender.notify_on_success = args.notify_success

    # Send notifications
    sender.run(status=args.status, notify_channels=notify_channels)


if __name__ == "__main__":
    main()
