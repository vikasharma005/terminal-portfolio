import '@commands/welcome.css';

import type { ComponentCommand } from '@commands';
import { i18n } from '@locale';
import { params } from '@nanostores/i18n';
import { useStore } from '@nanostores/preact';
import package_ from '@package';
import type { FunctionalComponent } from 'preact';

const messages = i18n('welcome', {
	author: '— Vikas Sharma',
	available_commands: 'To see a list of available commands, type ',
	quote: '"The future belongs to those who understand that data is the new oil, and AI is the refinery."',
	title: params<{ version: string }>('Welcome to my command line portfolio. (Version {version})'),
});

const Welcome: FunctionalComponent = () => {
	const t = useStore(messages);

	return (
		<div
			className='hero'
			data-testid='welcome'>
			<div className='info-section'>
				<p>{t.title({ version: package_.version })}</p>
				<span>----</span>
				<p
					style={{
						color: 'var(--color-primary)',
						fontStyle: 'italic',
						margin: '1rem 0',
						textAlign: 'left',
					}}>
					{t.quote}
				</p>
				{/* <p
					style={{
						color: 'var(--color-text-200)',
						fontSize: '0.9rem',
						marginBottom: '1rem',
						textAlign: 'left',
					}}>
					{t.author}
				</p> */}
				<span>----</span>
				<p>
					{t.available_commands}`<span className='command'>help</span>`.
				</p>
			</div>
			<div className='illu-section'>
				<pre className='img'>
					{`____    ___ .__   __                      
\\   \\  /  / |__| |  | __ _____      ______
 \\   \\/  /  |  | |  |/ / \\__  \\    /  ___/
  \\     /   |  | |    <   / __ \\_  \\___ \\ 
   \\___/    |__| |__|_ \\ (____  / /____  >
                      \\/      \\/       \\/ `}
				</pre>
			</div>
		</div>
	);
};

const WelcomeCommand: ComponentCommand = {
	command: 'welcome',
	component: Welcome,
};

export default WelcomeCommand;
