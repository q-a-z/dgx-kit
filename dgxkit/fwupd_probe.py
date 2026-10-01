"""Ask fwupd, over the system D-Bus, which firmware devices this machine has and which updates it knows of.

Read only: it lists devices and the updates on offer, and installs nothing. It runs in a short-lived container
(see system_info.probe_firmware) because the dashboard's own container can't reach the system bus.
Prints one JSON document.
"""
from __future__ import annotations

import json

UPDATABLE = 1 << 1  # FWUPD_DEVICE_FLAG_UPDATABLE


def main() -> None:
    from jeepney import DBusAddress, MessageType, new_method_call
    from jeepney.io.blocking import open_dbus_connection

    conn = open_dbus_connection(bus="SYSTEM")
    fwupd = DBusAddress("/", bus_name="org.freedesktop.fwupd", interface="org.freedesktop.fwupd")

    def call(member: str, signature: str = "", body: tuple = ()):
        reply = conn.send_and_get_reply(new_method_call(fwupd, member, signature, body), timeout=30)
        return None if reply.header.message_type == MessageType.error else reply.body[0]  # an error means "nothing to report"

    def plain(d: dict) -> dict:
        return {k: v[1] for k, v in d.items()}

    out = []
    for raw in call("GetDevices") or []:
        d = plain(raw)
        updatable = bool(int(d.get("Flags", 0)) & UPDATABLE)
        updates = []
        if updatable:
            for r in call("GetUpgrades", "s", (d["DeviceId"],)) or []:
                u = plain(r)
                updates.append({"version": u.get("Version"), "summary": u.get("Summary"), "remote": u.get("RemoteId")})
        out.append({"id": d.get("DeviceId"), "name": d.get("Name"), "version": d.get("Version"), "vendor": d.get("Vendor"),
                    "summary": d.get("Summary"), "plugin": d.get("Plugin"), "updatable": updatable, "updates": updates})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
