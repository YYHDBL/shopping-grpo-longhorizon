from html.parser import HTMLParser
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen


def _parse_action(action):
    if "[" not in action or not action.endswith("]"):
        return action, ""
    name, rest = action.split("[", 1)
    return name, rest[:-1]


class _ShopPageParser(HTMLParser):
    def __init__(self, base_url):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url.rstrip("/")
        self.text = []
        self.action_urls = {}
        self.current_form_action = None
        self.current_button_action = None
        self.current_link_href = None
        self.capture_reward = False
        self.reward_text = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form":
            action = attrs.get("action")
            self.current_form_action = urljoin(self.base_url, action) if action else None
        elif tag == "button":
            self.current_button_action = self.current_form_action
        elif tag == "a":
            href = attrs.get("href")
            self.current_link_href = urljoin(self.base_url, href) if href else None
        elif tag == "input" and attrs.get("type") == "radio":
            value = attrs.get("value")
            data_url = attrs.get("data-url")
            if value and data_url:
                self.action_urls[value] = urljoin(self.base_url, data_url)
        elif attrs.get("id") == "reward":
            self.capture_reward = True

    def handle_endtag(self, tag):
        if tag == "form":
            self.current_form_action = None
        elif tag == "button":
            self.current_button_action = None
        elif tag == "a":
            self.current_link_href = None
        elif tag == "div" and self.capture_reward:
            self.capture_reward = False

    def handle_data(self, data):
        value = data.strip()
        if not value:
            return
        self.text.append(value)
        if self.capture_reward:
            self.reward_text.append(value)
        if self.current_button_action:
            self.action_urls[value] = self.current_button_action
        if self.current_link_href:
            self.action_urls[value] = self.current_link_href


class ShopHttpEnv:
    def __init__(self, base_url="http://127.0.0.1:7001", timeout=60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session_id = None
        self.state = {}

    def reset(self, task_id):
        self.session_id = f"fixed_{int(task_id)}"
        return self._fetch(f"{self.base_url}/{self.session_id}")

    def step(self, action):
        url = self.build_action_url(action)
        if url is None:
            return {
                "observation": f"Error: action is not available: {action}",
                "html": "",
                "url": "",
                "status_code": 0,
                "reward": 0.0,
                "done": False,
                "available_actions": self.state.get("available_actions", []),
                "action_urls": self.state.get("action_urls", {}),
            }
        return self._fetch(url)

    def build_action_url(self, action):
        name, arg = _parse_action(action)
        if name == "search":
            keywords = quote(repr([arg]))
            return f"{self.base_url}/search_results/{self.session_id}/{keywords}/1"
        if name == "click":
            return self.state.get("action_urls", {}).get(arg)
        return None

    def parse_observation(self, html):
        parser = _ShopPageParser(self.base_url)
        parser.feed(html)
        reward = 0.0
        for token in parser.reward_text:
            try:
                reward = float(token)
                break
            except ValueError:
                pass
        return {
            "observation": " ".join(parser.text),
            "available_actions": list(parser.action_urls),
            "action_urls": parser.action_urls,
            "reward": reward,
            "done": reward > 0.0,
        }

    def _fetch(self, url):
        request = Request(url, headers={"User-Agent": "shopping-grpo-adapter/0.1"})
        with urlopen(request, timeout=self.timeout) as response:
            html = response.read().decode("utf-8", errors="replace")
            parsed = self.parse_observation(html)
            self.state = {
                **parsed,
                "html": html,
                "url": response.geturl(),
                "status_code": response.status,
            }
            return self.state
