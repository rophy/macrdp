use ironrdp_core::{decode, impl_as_any};
use ironrdp_dvc::{DvcMessage, DvcProcessor, DvcServerProcessor};
use ironrdp_pdu::{decode_err, PduResult};
use tracing::debug;

use crate::pdu::{DisplayControlCapabilities, DisplayControlMonitorLayout, DisplayControlPdu};
use crate::CHANNEL_NAME;

pub trait DisplayControlHandler: Send {
    fn monitor_layout(&self, layout: DisplayControlMonitorLayout) {
        debug!(?layout);
    }
}

/// A server for the Display Control Virtual Channel.
pub struct DisplayControlServer {
    handler: Box<dyn DisplayControlHandler>,
    max_num_monitors: u32,
}

impl DisplayControlServer {
    /// Create a new DisplayControlServer.
    pub fn new(handler: Box<dyn DisplayControlHandler>) -> Self {
        Self {
            handler,
            max_num_monitors: 1,
        }
    }

    /// Set the maximum number of monitors advertised to the client.
    pub fn with_max_monitors(mut self, max_num_monitors: u32) -> Self {
        self.max_num_monitors = max_num_monitors;
        self
    }
}

impl_as_any!(DisplayControlServer);

impl DvcProcessor for DisplayControlServer {
    fn channel_name(&self) -> &str {
        CHANNEL_NAME
    }

    fn start(&mut self, _channel_id: u32) -> PduResult<Vec<DvcMessage>> {
        let pdu: DisplayControlPdu =
            DisplayControlCapabilities::new(self.max_num_monitors, 3840, 2400)
                .map_err(|e| decode_err!(e))?
                .into();

        Ok(vec![Box::new(pdu)])
    }

    fn process(&mut self, _channel_id: u32, payload: &[u8]) -> PduResult<Vec<DvcMessage>> {
        match decode(payload).map_err(|e| decode_err!(e))? {
            DisplayControlPdu::MonitorLayout(layout) => self.handler.monitor_layout(layout),
            DisplayControlPdu::Caps(caps) => {
                debug!(?caps);
            }
        }
        Ok(Vec::new())
    }
}

impl DvcServerProcessor for DisplayControlServer {}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestHandler;
    impl DisplayControlHandler for TestHandler {}

    #[test]
    fn default_max_monitors_is_one() {
        let mut server = DisplayControlServer::new(Box::new(TestHandler));
        let messages = server.start(1).unwrap();
        assert_eq!(messages.len(), 1);

        let encoded = ironrdp_core::encode_vec(messages[0].as_ref()).unwrap();
        let pdu: DisplayControlPdu = decode(&encoded).unwrap();
        match pdu {
            DisplayControlPdu::Caps(caps) => {
                assert_eq!(caps.max_monitor_area(), 3840 * 2400);
            }
            _ => panic!("expected Caps PDU"),
        }
    }

    #[test]
    fn with_max_monitors_changes_caps() {
        let mut server = DisplayControlServer::new(Box::new(TestHandler))
            .with_max_monitors(4);
        let messages = server.start(1).unwrap();
        let encoded = ironrdp_core::encode_vec(messages[0].as_ref()).unwrap();
        let pdu: DisplayControlPdu = decode(&encoded).unwrap();
        match pdu {
            DisplayControlPdu::Caps(caps) => {
                assert_eq!(caps.max_monitor_area(), 3840 * 2400 * 4);
            }
            _ => panic!("expected Caps PDU"),
        }
    }

    #[test]
    fn channel_name_is_display_control() {
        let server = DisplayControlServer::new(Box::new(TestHandler));
        assert_eq!(server.channel_name(), CHANNEL_NAME);
    }

    #[test]
    fn process_monitor_layout_calls_handler() {
        use crate::pdu::{DisplayControlMonitorLayout, MonitorLayoutEntry};
        use ironrdp_core::encode_vec;

        let primary = MonitorLayoutEntry::new_primary(1920, 1080).unwrap();
        let layout = DisplayControlMonitorLayout::new(&[primary]).unwrap();
        let pdu = DisplayControlPdu::MonitorLayout(layout);
        let payload = encode_vec(&pdu).unwrap();

        let mut server = DisplayControlServer::new(Box::new(TestHandler));
        let result = server.process(1, &payload);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
}
