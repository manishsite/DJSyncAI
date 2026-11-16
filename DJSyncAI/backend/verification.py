import re
from typing import Tuple, Optional, Set

# Strict regex patterns for explicitly accepted extended version descriptors
ACCEPTED_EXTENDED_PATTERNS = [
    r"\bextended\s+mix\b",
    r"\bextended\s+edit\b",
    r"\bextended\s+club\s+mix\b",
    r"\bextended\s+club\s+version\b",
    r"\bextended\s+version\b",
    r"\bextended\s+remix\b",
    r"\bextended\s+dub\b",
    r"\bextended\s+vocal\s+mix\b",
    r"\bextended\s+instrumental\b",
    r"\bclub\s+extended\s+mix\b",
    r"\bextended\s+vip\b",
    r"\bextended\b"  # When in mix context e.g. (Extended)
]

# Strict regex patterns for explicitly rejected versions
REJECTED_VERSION_PATTERNS = [
    r"\bradio\s+edit\b",
    r"\bradio\s+mix\b",
    r"\bradio\s+version\b",
    r"\bshort\s+edit\b",
    r"\bshort\s+mix\b",
    r"\bsingle\s+edit\b",
    r"\bsingle\s+version\b",
    r"\bintro\s+edit\b",
    r"\boutro\s+edit\b",
    r"\bdj\s+intro\b(?!.*extended)",
    r"\bquick\s+hit\b",
    r"\bcool\s+down\s+edit\b",
    r"\btransition\s+edit\b",
    r"\bacoustic\b",
    r"\blive\b(?!.*extended)",
    r"\bspotify\s+edit\b",
    r"\btiktok\s+edit\b",
    r"\bmini\s+mix\b"
]

class ExtendedMixVerifier:
    """
    Strict Verification & Filtering Engine for DJSyncAI.
    Ensures that ONLY verified Extended Mixes / Extended Edits enter the candidate pool.
    """

    def __init__(self):
        self.seen_signatures: Set[str] = set()

    def reset_deduplication(self):
        self.seen_signatures.clear()

    @staticmethod
    def normalize_string(text: str) -> str:
        """Strip non-alphanumeric characters and lowercase for strict comparison."""
        if not text:
            return ""
        return re.sub(r"[^a-zA-Z0-9\s]", " ", text).lower().strip()

    @classmethod
    def extract_mix_label(cls, full_title: str, mix_field: Optional[str] = "") -> str:
        """
        Extract the explicit mix / version string from title parentheses, brackets, or mix field.
        e.g., 'Cola (Extended Mix)' -> 'Extended Mix'
        """
        candidates = []
        if mix_field and mix_field.strip():
            candidates.append(mix_field.strip())

        # Check for parentheses or brackets in title: "Song Name (Extended Mix)" or "[Extended Mix]"
        matches = re.findall(r"[\(\[](.*?)[\)\]]", full_title)
        for m in matches:
            candidates.append(m.strip())

        # Check for hyphen separation: "Song Name - Extended Mix"
        if " - " in full_title:
            parts = full_title.split(" - ")
            candidates.append(parts[-1].strip())

        for cand in candidates:
            cand_lower = cand.lower()
            for pattern in ACCEPTED_EXTENDED_PATTERNS:
                if re.search(pattern, cand_lower):
                    # Canonicalize representation
                    if "extended club mix" in cand_lower:
                        return "Extended Club Mix"
                    if "extended remix" in cand_lower:
                        return "Extended Remix"
                    if "extended dub" in cand_lower:
                        return "Extended Dub"
                    if "extended edit" in cand_lower:
                        return "Extended Edit"
                    return "Extended Mix"

        return mix_field.strip() if mix_field else "Original Mix"

    @classmethod
    def verify(cls, artist: str, title: str, mix_name: str) -> Tuple[bool, str, str]:
        """
        Evaluate candidate against strict Extended Mix criteria.
        Returns:
            (is_verified: bool, canonical_version_type: str, reason_or_evidence: str)
        """
        combined = f"{title} {mix_name}".lower()

        # 1. Check for immediate rejection patterns
        for pattern in REJECTED_VERSION_PATTERNS:
            match = re.search(pattern, combined)
            if match:
                matched_label = match.group(0).title()
                return False, "Rejected", f"Excluded: explicitly identified as a shortened/radio edit ({matched_label})"

        # 2. Check for explicit Original Mix without extended label
        if re.search(r"\boriginal\s+mix\b", combined) and not any(re.search(p, combined) for p in ACCEPTED_EXTENDED_PATTERNS):
            return False, "Rejected", "Excluded: standard Original Mix without explicit Extended Mix verification"

        if re.search(r"\balbum\s+version\b", combined) and not any(re.search(p, combined) for p in ACCEPTED_EXTENDED_PATTERNS):
            return False, "Rejected", "Excluded: standard Album Version without explicit Extended Mix verification"

        # 3. Check for explicit accepted extended labels
        for pattern in ACCEPTED_EXTENDED_PATTERNS:
            if re.search(pattern, combined):
                canonical_label = cls.extract_mix_label(title, mix_name)
                return True, canonical_label, f"Verified as authentic {canonical_label} via catalog metadata"

        # 4. If no explicit extended mix label is present, REJECT. Never guess based on duration.
        return False, "Unverified", "Excluded: lacks explicit Extended Mix / Extended Edit identification"

    def is_duplicate(self, artist: str, title: str) -> bool:
        """
        Ensure deduplication by normalized Artist and Title.
        Treats remix versions distinctly if specified.
        """
        # Clean title of mix parentheticals for core identity
        core_title = re.sub(r"[\(\[].*?[\)\]]", "", title)
        core_title = re.sub(r"\s-\s.*$", "", core_title)
        
        norm_artist = self.normalize_string(artist)
        norm_title = self.normalize_string(core_title)

        signature = f"{norm_artist}___{norm_title}"
        if signature in self.seen_signatures:
            return True
        self.seen_signatures.add(signature)
        return False
