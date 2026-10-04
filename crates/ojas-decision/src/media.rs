//! Images a request carries: where they come from, how many, and how they are
//! decoded.
//!
//! A request gives images in its `images` array, or as `image_url` parts of the
//! chat messages its state holds (`state` itself, or `state.messages`). Both are
//! `data:image/...;base64,...` URLs. They reach the prompt in that order: the
//! `images` array first, then each message's parts. The parts are taken out of the
//! messages, so the state the template renders holds only their text.

use super::json::Json;
use super::{invalid, InvalidRequest};
use anyhow::Result;

/// Most images in one request.
pub const MAX_IMAGES: usize = 8;

/// A decoded image: interleaved RGB, 8 bits per channel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

impl Image {
    /// Decode a `data:image/...;base64,...` URL. Errors are [`InvalidRequest`].
    pub fn from_data_url(url: &str) -> Result<Image> {
        let bad = || invalid("images must be data URLs (data:image/...;base64,...)");
        let rest = url.strip_prefix("data:image/").ok_or_else(bad)?;
        let (_, payload) = rest.split_once(";base64,").ok_or_else(bad)?;
        let bytes = decode_base64(payload).ok_or_else(|| invalid("an image's base64 data is malformed"))?;
        Image::decode(&bytes)
    }

    /// Decode an encoded image (PNG, JPEG, ...). Errors are [`InvalidRequest`].
    pub fn decode(bytes: &[u8]) -> Result<Image> {
        let (width, height, rgb) = ojas_cpu::vit_preprocess::decode_rgb8(bytes)
            .map_err(|e| InvalidRequest(format!("an image could not be decoded: {e}")))?;
        Ok(Image { width, height, rgb })
    }
}

/// Take the images out of a request: the `images` array, then the `image_url`
/// parts of the messages in `state`. Returns the state without those parts.
pub(super) fn take_images(body: &Json, state: &Json) -> Result<(Json, Vec<Image>)> {
    let mut images = Vec::new();
    let mut add = |url: &Json| -> Result<()> {
        let url = url.as_str().ok_or_else(|| invalid("images must be data URLs (data:image/...;base64,...)"))?;
        if images.len() == MAX_IMAGES {
            return Err(invalid(format!("too many images, the maximum is {MAX_IMAGES}")));
        }
        images.push(Image::from_data_url(url)?);
        Ok(())
    };
    match body.get("images") {
        None | Some(Json::Null) => {}
        Some(Json::Array(urls)) => for url in urls { add(url)?; },
        Some(_) => return Err(invalid("\"images\" must be an array")),
    }
    let state = match state {
        Json::Array(messages) => Json::Array(take_from_messages(messages, &mut add)?),
        Json::Object(kv) if matches!(state.get("messages"), Some(Json::Array(_))) => Json::Object(
            kv.iter().map(|(k, v)| Ok((k.clone(), match (k.as_str(), v) {
                ("messages", Json::Array(messages)) => Json::Array(take_from_messages(messages, &mut add)?),
                _ => v.clone(),
            }))).collect::<Result<_>>()?,
        ),
        other => other.clone(),
    };
    Ok((state, images))
}

/// The messages with their `image_url` parts handed to `add` and removed.
fn take_from_messages(messages: &[Json], add: &mut impl FnMut(&Json) -> Result<()>) -> Result<Vec<Json>> {
    messages.iter().map(|msg| {
        let Some(Json::Array(parts)) = msg.get("content") else { return Ok(msg.clone()) };
        let mut kept = Vec::with_capacity(parts.len());
        for part in parts {
            match (part.get("type").and_then(Json::as_str), part.get("image_url")) {
                (Some("image_url"), Some(url)) => add(url.get("url").unwrap_or(url))?,
                _ => kept.push(part.clone()),
            }
        }
        let Json::Object(kv) = msg else { unreachable!("a message with content is an object") };
        Ok(Json::Object(kv.iter().map(|(k, v)| (k.clone(), if k == "content" { Json::Array(kept.clone()) } else { v.clone() })).collect()))
    }).collect()
}

/// Standard base64 (RFC 4648), with or without padding; `None` on any other byte.
fn decode_base64(text: &str) -> Option<Vec<u8>> {
    fn value(c: u8) -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        } as u32)
    }
    let body = text.trim_end_matches('=').as_bytes();
    if body.len() % 4 == 1 || text.len() - body.len() > 2 { return None; }
    let mut out = Vec::with_capacity(body.len() * 3 / 4);
    for group in body.chunks(4) {
        let mut acc = 0u32;
        for &c in group { acc = acc << 6 | value(c)?; }
        acc <<= 6 * (4 - group.len()) as u32;
        out.extend_from_slice(&acc.to_be_bytes()[1..group.len()]);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2x1 PNG: one red pixel, one blue.
    const PNG_2X1: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAADUlEQVR4nGP4zwAE/wEHAAH/4iOeWQAAAABJRU5ErkJggg==";

    #[test]
    fn base64_decodes_with_and_without_padding() {
        assert_eq!(decode_base64("aGVsbG8="), Some(b"hello".to_vec()));
        assert_eq!(decode_base64("aGVsbG8"), Some(b"hello".to_vec()));
        assert_eq!(decode_base64("aGk="), Some(b"hi".to_vec()));
        assert_eq!(decode_base64(""), Some(Vec::new()));
        assert_eq!(decode_base64("a"), None);
        assert_eq!(decode_base64("aG!s"), None);
    }

    #[test]
    fn data_urls_decode_to_rgb() {
        let image = Image::from_data_url(&format!("data:image/png;base64,{PNG_2X1}")).unwrap();
        assert_eq!((image.width, image.height, image.rgb), (2, 1, vec![255, 0, 0, 0, 0, 255]));
        for url in ["https://example.com/a.png", "data:text/plain;base64,aGk=", "data:image/png,raw"] {
            assert!(Image::from_data_url(url).unwrap_err().downcast_ref::<InvalidRequest>().is_some(), "{url}");
        }
    }

    #[test]
    fn images_come_from_the_array_then_from_message_parts() {
        let url = format!("data:image/png;base64,{PNG_2X1}");
        let body = Json::parse(&format!(r#"{{"images": ["{url}"]}}"#)).unwrap();
        let state = Json::parse(&format!(r#"{{"messages": [{{"role": "user", "content": [
            {{"type": "image_url", "image_url": {{"url": "{url}"}}}}, {{"type": "text", "text": "hi"}}]}}], "plan": "pro"}}"#)).unwrap();
        let (state, images) = take_images(&body, &state).unwrap();
        assert_eq!(images.len(), 2);
        assert_eq!(state.to_python(false),
            r#"{"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], "plan": "pro"}"#);
    }

    #[test]
    fn too_many_or_malformed_images_are_refused() {
        let url = format!("\"data:image/png;base64,{PNG_2X1}\"");
        let nine = Json::parse(&format!("{{\"images\": [{}]}}", [url.as_str(); MAX_IMAGES + 1].join(", "))).unwrap();
        for body in [nine, Json::parse(r#"{"images": "x"}"#).unwrap(), Json::parse(r#"{"images": [3]}"#).unwrap()] {
            let err = take_images(&body, &Json::Str("s".into())).unwrap_err();
            assert!(err.downcast_ref::<InvalidRequest>().is_some(), "{err}");
        }
    }
}
